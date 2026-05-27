import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "jaide-v40-distributed-training"
GPU_SPEC = "B200:8"
DATA_VOLUME_NAME = "jaide-training-data"
CHECKPOINT_VOLUME_NAME = "jaide-checkpoints"

DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
PROJECT_MOUNT_PATH = Path("/jaide")

DATASET_DIR = DATA_MOUNT_PATH / "dataset"
DATASET_FILE = DATASET_DIR / "train.jsonl"
DATASET_METADATA_FILE = DATASET_DIR / "metadata.json"

BINARY_PATH = PROJECT_MOUNT_PATH / "zig-out" / "bin" / "jaide-distributed-futhark"

CPU_REQUEST = 64.0
CPU_LIMIT = 80.0
MEMORY_REQUEST_MB = 262144
MEMORY_LIMIT_MB = 262144
EPHEMERAL_DISK_MB = 3145728
TIMEOUT_SECONDS = 86400

LOCAL_PROJECT_DIR = (Path(__file__).resolve().parent / "../..").resolve()

IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "zig-cache",
    ".pythonlibs",
    ".cache",
    ".upm",
    "__pycache__",
    ".local",
    ".replit",
    "*.bin",
]

app = modal.App(APP_NAME)

jaide_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages git curl xz-utils build-essential wget ca-certificates",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "zstandard", "datasets", "huggingface_hub", "hf_xet")
    .run_commands(
        "mkdir -p /opt",
        "curl -sL https://ziglang.org/download/0.14.1/zig-linux-x86_64-0.14.1.tar.xz | tar -xJ -C /opt",
        "ln -sf /opt/zig-linux-x86_64-0.14.1/zig /usr/local/bin/zig",
        "zig version",
    )
    .env(
        {
            "PATH": "/opt/zig-linux-x86_64-0.14.1:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": "/data/hf_home",
            "HF_DATASETS_CACHE": "/data/hf_datasets_cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(PROJECT_MOUNT_PATH),
        ignore=IGNORE_PATTERNS,
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


def _run_checked(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as e:
        return 127, "", str(e)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_json_file(path: Path, value: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _extract_text_from_row(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("text", "content", "sentence", "article"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for val in row.values():
        if isinstance(val, str) and len(val.strip()) > 50:
            return val.strip()
    return ""


def download_finephrase_to_jsonl(volume: modal.Volume) -> Tuple[str, int, int]:
    from datasets import load_dataset

    _ensure_dir(DATASET_DIR)

    if DATASET_FILE.is_file() and DATASET_FILE.stat().st_size > 0:
        size = int(DATASET_FILE.stat().st_size)
        metadata = _read_json_file(DATASET_METADATA_FILE)
        if metadata is not None and int(metadata.get("dataset_size", -1)) == size and int(metadata.get("line_count", 0)) > 0:
            return str(DATASET_FILE), size, int(metadata["line_count"])
        line_count = _count_lines(DATASET_FILE)
        if line_count > 0:
            _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
            volume.commit()
            return str(DATASET_FILE), size, line_count
        DATASET_FILE.unlink()

    tmp_file = DATASET_FILE.with_suffix(".jsonl.tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    ds = load_dataset("HuggingFaceFW/finephrase", split="train")

    line_count = 0
    with open(tmp_file, "w", encoding="utf-8") as f_out:
        for row in ds:
            text = _extract_text_from_row(row)
            if text and len(text) > 20:
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                line_count += 1

    if line_count <= 0:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError("Dataset conversion produced zero usable samples")

    tmp_file.replace(DATASET_FILE)
    size = int(DATASET_FILE.stat().st_size)
    _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
    volume.commit()
    return str(DATASET_FILE), size, line_count


def _build_zig_gpu(project_dir: str) -> None:
    if BINARY_PATH.is_file():
        return
    rc, out, err = _run_checked(
        ["zig", "build", "-Dgpu=true", "-Doptimize=ReleaseFast"],
        cwd=project_dir,
    )
    if rc != 0:
        raise RuntimeError(f"GPU build failed with exit code {rc}: {(err or out)[-8000:]}")
    if not BINARY_PATH.is_file():
        raise FileNotFoundError(f"GPU binary not found at {BINARY_PATH}")
    BINARY_PATH.chmod(0o755)


def _detect_gpus() -> Tuple[int, str]:
    try:
        p = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    except FileNotFoundError as e:
        return 0, str(e)
    output = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return len(lines), output


def _expected_gpu_count() -> int:
    if ":" not in GPU_SPEC:
        return 1
    try:
        return int(GPU_SPEC.rsplit(":", 1)[1])
    except ValueError:
        return 1


def _extract_loss(stdout: str) -> Optional[float]:
    if not stdout:
        return None
    loss_value = None
    pattern = re.compile(r"\b(?:loss|train_loss|training_loss)\b[^0-9+\-]*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", re.IGNORECASE)
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            try:
                loss_value = float(match.group(1))
            except ValueError:
                continue
    return loss_value


def _read_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    byte_count = min(size, max_chars * 4)
    with open(path, "rb") as f:
        if size > byte_count:
            f.seek(size - byte_count)
        data = f.read()
    return data.decode("utf-8", errors="replace")[-max_chars:]


@app.function(
    image=jaide_image,
    gpu=GPU_SPEC,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    ephemeral_disk=EPHEMERAL_DISK_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
    },
)
def train_rank(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    epochs: int,
    model_dim: int,
    num_layers: int,
    local_batch_size: int,
    dataset_path: str,
) -> Dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()

    gpu_count, gpu_list = _detect_gpus()
    expected_gpus = _expected_gpu_count()
    if gpu_count < 1:
        raise RuntimeError(f"No NVIDIA GPUs detected: {gpu_list}")
    if gpu_count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs from {GPU_SPEC}, detected {gpu_count}: {gpu_list}")

    os.chdir(str(PROJECT_MOUNT_PATH))
    _ensure_dir(CHECKPOINT_MOUNT_PATH)

    _build_zig_gpu(str(PROJECT_MOUNT_PATH))

    env = os.environ.copy()
    env["WORLD_SIZE"] = str(world_size)
    env["RANK"] = str(rank)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["JAIDE_EPOCHS"] = str(epochs)
    env["JAIDE_DATASET"] = dataset_path
    env["JAIDE_MODEL_DIM"] = str(model_dim)
    env["JAIDE_LAYERS"] = str(num_layers)
    env["JAIDE_BATCH_SIZE"] = str(local_batch_size)
    env["CUDA_VISIBLE_DEVICES"] = str(rank % gpu_count)

    logs_dir = Path("/tmp/jaide_training_logs")
    _ensure_dir(logs_dir)
    stdout_path = logs_dir / f"rank_{rank:03d}.stdout.log"
    stderr_path = logs_dir / f"rank_{rank:03d}.stderr.log"

    start_time = time.time()
    return_code = -1
    stdout_tail = ""
    stderr_tail = ""
    timed_out = False

    with open(stdout_path, "w", encoding="utf-8", errors="replace") as stdout_file, open(stderr_path, "w", encoding="utf-8", errors="replace") as stderr_file:
        try:
            proc = subprocess.Popen(
                [str(BINARY_PATH)],
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                cwd=str(PROJECT_MOUNT_PATH),
                preexec_fn=os.setsid,
            )
            try:
                return_code = proc.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, 15)
                    proc.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, 9)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                return_code = -9
        except FileNotFoundError as e:
            stderr_file.write(str(e))
            return_code = 127

    elapsed_time = float(time.time() - start_time)
    stdout_tail = _read_tail(stdout_path)
    stderr_tail = _read_tail(stderr_path)

    loss = _extract_loss(stdout_tail) if rank == 0 else None

    return {
        "rank": rank,
        "return_code": int(return_code),
        "timed_out": bool(timed_out),
        "elapsed_seconds": elapsed_time,
        "loss": loss,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.function(
    image=jaide_image,
    cpu=8.0,
    memory=16384,
    timeout=TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
    },
)
def orchestrate_training(
    epochs: int = 20,
    model_dim: int = 512,
    num_layers: int = 16,
    local_batch_size: int = 4,
    world_size: int = 8,
) -> Dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()

    dataset_path, dataset_size, sample_count = download_finephrase_to_jsonl(data_volume)
    if sample_count <= 0:
        raise RuntimeError("Dataset contains zero samples")

    master_addr = "localhost"
    master_port = "29500"

    print(f"Starting distributed training with {world_size} ranks")
    print(f"Model: dim={model_dim}, layers={num_layers}, batch={local_batch_size}")
    print(f"Dataset: {sample_count} samples, {dataset_size / 1e6:.2f} MB")
    print(f"Epochs: {epochs}")

    rank_futures = []
    for rank in range(world_size):
        future = train_rank.spawn(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            epochs=epochs,
            model_dim=model_dim,
            num_layers=num_layers,
            local_batch_size=local_batch_size,
            dataset_path=dataset_path,
        )
        rank_futures.append(future)

    results = []
    for future in rank_futures:
        try:
            result = future.get(timeout=TIMEOUT_SECONDS + 300)
            results.append(result)
        except Exception as e:
            results.append({
                "rank": len(results),
                "return_code": -1,
                "error": str(e),
            })

    successful_ranks = [r for r in results if int(r.get("return_code", -1)) == 0 and not r.get("timed_out", False)]
    rank_0_result = next((r for r in results if r.get("rank") == 0), None)

    final_loss = rank_0_result.get("loss") if rank_0_result else 0.0
    completed_epochs = epochs if len(successful_ranks) == world_size else 0
    status = "completed" if len(successful_ranks) == world_size else "failed"

    _write_json_file(
        CHECKPOINT_MOUNT_PATH / "training_complete.json",
        {
            "status": status,
            "total_epochs": epochs,
            "completed_epochs": completed_epochs,
            "final_loss": final_loss,
            "dataset_path": str(dataset_path),
            "dataset_size_mb": float(dataset_size) / 1e6,
            "sample_count": sample_count,
            "gpu_config": f"{world_size}x NVIDIA B200",
            "model_dim": model_dim,
            "num_layers": num_layers,
            "local_batch_size": local_batch_size,
            "world_size": world_size,
            "rank_results": results,
        },
    )

    checkpoint_volume.commit()

    return {
        "status": status,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "final_loss": final_loss,
        "dataset_size_mb": float(dataset_size) / 1e6,
        "sample_count": sample_count,
        "gpu_config": f"{world_size}x NVIDIA B200",
        "model_dim": model_dim,
        "num_layers": num_layers,
        "successful_ranks": len(successful_ranks),
        "rank_results": results,
    }


@app.local_entrypoint()
def main(
    epochs: int = 20,
    model_dim: int = 512,
    num_layers: int = 16,
    local_batch_size: int = 4,
    world_size: int = 8,
) -> None:
    result = orchestrate_training.remote(
        epochs=epochs,
        model_dim=model_dim,
        num_layers=num_layers,
        local_batch_size=local_batch_size,
        world_size=world_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
