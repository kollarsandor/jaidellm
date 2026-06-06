const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const dtf = @import("distributed/distributed_trainer_futhark.zig");
const DistributedTrainerFuthark = dtf.DistributedTrainerFuthark;
const TrainerConfig = dtf.TrainerConfig;
const nccl = @import("distributed/nccl_bindings.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const world_size = try std.process.getEnvVarOwned(allocator, "WORLD_SIZE");
    defer allocator.free(world_size);
    const world_size_val = try std.fmt.parseInt(usize, world_size, 10);

    const rank_str = try std.process.getEnvVarOwned(allocator, "RANK");
    defer allocator.free(rank_str);
    const rank = try std.fmt.parseInt(usize, rank_str, 10);

    const master_addr = try std.process.getEnvVarOwned(allocator, "MASTER_ADDR");
    defer allocator.free(master_addr);

    const master_port = try std.process.getEnvVarOwned(allocator, "MASTER_PORT");
    defer allocator.free(master_port);

    std.debug.print("============================================================\n", .{});
    std.debug.print("JAIDE v40 Distributed Training (Futhark GPU Acceleration)\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Rank: {d}/{d}\n", .{ rank, world_size_val });
    std.debug.print("Master: {s}:{s}\n", .{ master_addr, master_port });
    std.debug.print("GPU: NVIDIA B200 (192GB)\n", .{});
    std.debug.print("Precision: f16 (Futhark kernels)\n", .{});
    std.debug.print("NVLink: Enabled (NCCL P2P)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    var nccl_id_path_owned: ?[]u8 = null;
    const nccl_id_path: []const u8 = blk: {
        nccl_id_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_NCCL_ID_PATH") catch null;
        break :blk nccl_id_path_owned orelse "/tmp/jaide_nccl_id";
    };
    defer if (nccl_id_path_owned) |owned| allocator.free(owned);

    var nccl_ready_path_buf: [256]u8 = undefined;
    const nccl_ready_path = try std.fmt.bufPrint(&nccl_ready_path_buf, "{s}.ready", .{nccl_id_path});

    var nccl_id: nccl.ncclUniqueId = undefined;

    if (rank == 0) {
        const result = nccl.ncclGetUniqueId(&nccl_id);
        if (result != .ncclSuccess) {
            std.debug.print("Failed to generate NCCL ID\n", .{});
            return error.NCCLGetUniqueIdFailed;
        }

        std.fs.deleteFileAbsolute(nccl_id_path) catch {};
        std.fs.deleteFileAbsolute(nccl_ready_path) catch {};

        const id_file = try std.fs.createFileAbsolute(nccl_id_path, .{});
        try id_file.writeAll(std.mem.asBytes(&nccl_id));
        try id_file.sync();
        id_file.close();

        const ready_file = try std.fs.createFileAbsolute(nccl_ready_path, .{});
        try ready_file.writeAll("ready");
        try ready_file.sync();
        ready_file.close();

        std.debug.print("[Rank 0] Generated NCCL ID (file: {s})\n", .{nccl_id_path});
    } else {
        var attempts: usize = 0;
        while (attempts < 100) : (attempts += 1) {
            const ready_file = std.fs.openFileAbsolute(nccl_ready_path, .{}) catch {
                std.time.sleep(100 * std.time.ns_per_ms);
                continue;
            };
            ready_file.close();
            break;
        }

        if (attempts >= 100) {
            std.debug.print("[Rank {d}] Timeout waiting for NCCL ID from rank 0\n", .{rank});
            return error.NCCLIdTimeout;
        }

        const id_file = try std.fs.openFileAbsolute(nccl_id_path, .{});
        defer id_file.close();

        const bytes_read = try id_file.readAll(std.mem.asBytes(&nccl_id));
        if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
            std.debug.print("[Rank {d}] Failed to read NCCL ID (got {d} bytes, expected {d})\n", .{ rank, bytes_read, @sizeOf(nccl.ncclUniqueId) });
            return error.NCCLIdReadFailed;
        }

        std.debug.print("[Rank {d}] Loaded NCCL ID from rank 0\n", .{rank});
    }

    var coordinator = try GPUCoordinator.init(allocator, world_size_val, rank, nccl_id);
    defer coordinator.deinit();

    std.debug.print("[Rank {d}] GPU coordinator initialized\n", .{rank});

    var model_dim_str_owned: ?[]u8 = null;
    const model_dim_str: []const u8 = blk: {
        model_dim_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_MODEL_DIM") catch null;
        break :blk model_dim_str_owned orelse "2048";
    };
    defer if (model_dim_str_owned) |owned| allocator.free(owned);
    const model_dim = std.fmt.parseInt(usize, model_dim_str, 10) catch 2048;

    var num_layers_str_owned: ?[]u8 = null;
    const num_layers_str: []const u8 = blk: {
        num_layers_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_LAYERS") catch null;
        // 24 matches the baseline Modal config (model_dim=2048,layers=24 ≈ 50M params).
        // The Modal orchestrator always sets JAIDE_LAYERS explicitly so this
        // default only matters for local invocations.
        break :blk num_layers_str_owned orelse "24";
    };
    defer if (num_layers_str_owned) |owned| allocator.free(owned);
    const num_layers = std.fmt.parseInt(usize, num_layers_str, 10) catch 24;

    var local_batch_size_str_owned: ?[]u8 = null;
    const local_batch_size_str: []const u8 = blk: {
        local_batch_size_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_BATCH_SIZE") catch null;
        break :blk local_batch_size_str_owned orelse "4";
    };
    defer if (local_batch_size_str_owned) |owned| allocator.free(owned);
    const local_batch_size = std.fmt.parseInt(usize, local_batch_size_str, 10) catch 4;

    var epochs_env_owned: ?[]u8 = null;
    const epochs_env: []const u8 = blk: {
        epochs_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_EPOCHS") catch null;
        break :blk epochs_env_owned orelse "20";
    };
    defer if (epochs_env_owned) |owned| allocator.free(owned);

    const num_epochs = std.fmt.parseInt(usize, epochs_env, 10) catch 20;

    var lr_env_owned: ?[]u8 = null;
    const lr_str: []const u8 = blk: {
        lr_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_LEARNING_RATE") catch null;
        break :blk lr_env_owned orelse "0.0001";
    };
    defer if (lr_env_owned) |owned| allocator.free(owned);
    const learning_rate: f32 = std.fmt.parseFloat(f32, lr_str) catch 0.0001;

    var trainer_cfg: TrainerConfig = .{};
    trainer_cfg.learning_rate = learning_rate;

    var trainer = try DistributedTrainerFuthark.initWithConfig(
        allocator,
        &coordinator,
        model_dim,
        num_layers,
        local_batch_size,
        trainer_cfg,
    );
    defer trainer.deinit();

    std.debug.print("[Rank {d}] learning_rate={d}\n", .{ rank, learning_rate });

    std.debug.print("[Rank {d}] Futhark-accelerated trainer initialized (f16, model_dim={d}, layers={d})\n", .{ rank, model_dim, num_layers });

    var dataset_path_owned: ?[]u8 = null;
    const dataset_path: []const u8 = blk: {
        dataset_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_DATASET") catch null;
        break :blk dataset_path_owned orelse "/data/dataset/train.jsonl";
    };
    defer if (dataset_path_owned) |owned| allocator.free(owned);

    std.debug.print("[Rank {d}] Loading dataset from {s}\n", .{ rank, dataset_path });

    const samples = try trainer.loadDataset(dataset_path);
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Starting Futhark-accelerated training\n", .{});
        std.debug.print("Dataset: {d} samples (per rank)\n", .{samples.len});
        std.debug.print("Batch size: {d} (per rank)\n", .{local_batch_size});
        std.debug.print("Epochs: {d}\n", .{num_epochs});
        std.debug.print("GPU Memory: 100%% VRAM-resident (zero host copies)\n", .{});
        std.debug.print("NVLink: Enabled for gradient synchronization\n", .{});
        std.debug.print("============================================================\n\n", .{});
    }

    for (samples) |sample_text| {
        trainer.buildKnowledgeGraph(sample_text) catch |err| {
            std.debug.print("[Rank {d}] CREV processTextStream warning: {} (continuing)\n", .{ rank, err });
        };
    }

    if (coordinator.isRoot()) {
        std.debug.print("[Rank 0] Knowledge graph populated.\n", .{});
    }

    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        const start_time = std.time.milliTimestamp();

        const avg_loss = trainer.trainEpoch(samples) catch |err| {
            std.debug.print("[Rank {d}] trainEpoch ERROR (epoch={d}): {}\n", .{ rank, epoch + 1, err });
            return err;
        };

        const end_time = std.time.milliTimestamp();
        const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0;

        if (coordinator.isRoot()) {
            std.debug.print("[Epoch {d}/{d}] Loss: {d:.6} | Time: {d:.2}s\n", .{ epoch + 1, num_epochs, avg_loss, elapsed });

            {
                var dir_buf: [256]u8 = undefined;
                const dir_path = std.fmt.bufPrint(&dir_buf, "/checkpoints/epoch_{d:0>3}", .{epoch + 1}) catch "/checkpoints";
                // Ensure both /checkpoints (parent volume mount) and the
                // per-epoch subdirectory exist. makeDirAbsolute only creates
                // the leaf and silently fails (ENOENT) if the parent isn't
                // there. By making both we tolerate either fresh boots or
                // Modal mounts where the parent is pre-created.
                std.fs.makeDirAbsolute("/checkpoints") catch |e| switch (e) {
                    error.PathAlreadyExists => {},
                    else => std.debug.print("[Rank 0] makeDirAbsolute(/checkpoints) failed: {} (continuing)\n", .{e}),
                };
                std.fs.makeDirAbsolute(dir_path) catch |e| switch (e) {
                    error.PathAlreadyExists => {},
                    else => std.debug.print("[Rank 0] makeDirAbsolute({s}) failed: {} (continuing)\n", .{ dir_path, e }),
                };

                var checkpoint_path_buf: [256]u8 = undefined;
                const checkpoint_path = try std.fmt.bufPrint(
                    &checkpoint_path_buf,
                    "/checkpoints/epoch_{d:0>3}/model.ckpt",
                    .{epoch + 1},
                );

                try trainer.saveCheckpoint(checkpoint_path);
                std.debug.print("  Checkpoint saved: {s}\n", .{checkpoint_path});
            }
        }

        try coordinator.synchronize();
    }

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Futhark-accelerated training completed successfully!\n", .{});
        std.debug.print("Final model saved to /checkpoints/\n", .{});
        std.debug.print("============================================================\n", .{});
    }
}

================