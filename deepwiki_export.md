# JAIDE v40 — DeepWiki Export

> Automatically exported from [DeepWiki](https://deepwiki.com/kollarsandor/jaidellm)

---

## Table of Contents

1. [JAIDE v40 — Project Overview](#page-1)
2. [Getting Started — Build, Configuration & Entrypoints](#page-2)
3. [Architecture Overview — RSF, NSIR, and the Processing Pipeline](#page-3)
4. [Core Primitives](#page-4)
5. [Tensor System](#page-5)
6. [Memory Management](#page-6)
7. [I/O and Model Persistence](#page-7)
8. [Neural Processing — RSF and OFTB](#page-8)
9. [RSF — Reversible Scatter Flow Processor](#page-9)
10. [OFTB — Orthogonal Fractal Transform Block](#page-10)
11. [Tokenizer and Retrieval](#page-11)
12. [MGT — Morpheme-Guided Tokenizer](#page-12)
13. [SSI — Structured Sequence Index](#page-13)
14. [Ranker — Sequence Scoring and Candidate Evaluation](#page-14)
15. [NSIR — Quantum-Relational Graph System](#page-15)
16. [NSIR Core — Graph Structure and Quantum Operations](#page-16)
17. [Reasoning Orchestrator and Energy Minimization](#page-17)
18. [CREV Pipeline — Knowledge Extraction and Triplet Management](#page-18)
19. [Quantum Backend Integration](#page-19)
20. [Optimization and Training](#page-20)
21. [SFD Optimizer — Second-Order Training](#page-21)
22. [Distributed Training](#page-22)
23. [Cloud Training with Modal](#page-23)
24. [Hardware Acceleration Layer](#page-24)
25. [Futhark GPU Kernels](#page-25)
26. [CUDA Bindings and Accelerator Interface](#page-26)
27. [Clash RTL Components](#page-27)
28. [Inference Server and API](#page-28)
29. [InferenceServer — HTTP API and Request Lifecycle](#page-29)
30. [Verified Inference Engine and ZK Proofs](#page-30)
31. [Security, Safety, and Formal Verification](#page-31)
32. [Formal Verification and Security Proofs](#page-32)
33. [Safety, Obfuscation, and C API](#page-33)
34. [Glossary](#page-34)

---

<a id="page-1"></a>

# JAIDE v40 — Project Overview




JAIDE (v40) is a Large Language Model (LLM) built from the ground up on the **Reversible Scatter Flow (RSF)** paradigm. Unlike traditional Transformer or CNN architectures, JAIDE utilizes bijective coupling layers that enable O(1) memory backpropagation and a parameter-less Haar-wavelet mixing block known as the **OFTB**.

The system is engineered for high-performance execution across a spectrum of hardware, ranging from standard CPUs to multi-GPU B200 clusters and quantum relational graphs.

### The RSF Paradigm

The core of JAIDE is the Reversible Scatter Flow, which replaces traditional self-attention and MLP structures with cross-affine coupling.

*   **Bijectivity:** Every forward pass has an exact algebraic inverse, ensuring no information collapse during processing.
*   **Memory Efficiency:** Because the network is reversible, activations do not need to be cached for backpropagation. The system reconstructs inputs from outputs in-place.
*   **Minimalist Primitives:** The architecture eliminates `softmax`, `attention`, `ReLU`, and `LayerNorm`, relying exclusively on learned scale and translation tensors.

### System Architecture and Data Flow

The following diagram bridges the conceptual "Natural Language Space" with the "Code Entity Space," showing how a request flows through the primary Zig components.

**Diagram: Request Processing Pipeline**
![Request Processing Pipeline](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgIElucHV0WyJVc2VyIFByb21wdCAoVGV4dCkiXQogICAgICAgIE91dHB1dFsiR2VuZXJhdGVkIFJlc3BvbnNlIChUZXh0KSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdHkgU3BhY2UiCiAgICAgICAgU2VydmVyWyJqYWlkZS1pbmZlcmVuY2Utc2VydmVyIChzcmMvaW5mZXJlbmNlX3NlcnZlcl9tYWluLnppZykiXQogICAgICAgIFRva2VuaXplclsiTUdUIChzcmMvdG9rZW5pemVyL21ndC56aWcpIl0KICAgICAgICBSU0ZbIlJTRkxheWVyIChzcmMvcHJvY2Vzc29yL3JzZi56aWcpIl0KICAgICAgICBTU0lbIlNTSSBJbmRleCAoc3JjL3Rva2VuaXplci9zc2kuemlnKSJdCiAgICAgICAgTlNJUlsiTlNJUiBHcmFwaCAoc3JjL25zaXIvY29yZS56aWcpIl0KICAgICAgICAKICAgICAgICBJbnB1dCAtLT4gU2VydmVyCiAgICAgICAgU2VydmVyIC0tPiBUb2tlbml6ZXIKICAgICAgICBUb2tlbml6ZXIgLS0+fCJUZW5zb3IifCBSU0YKICAgICAgICBSU0YgPC0tPnwiUmV0cmlldmUvSW5kZXgifCBTU0kKICAgICAgICBSU0YgPC0tPnwiUmVhc29uaW5nInwgTlNJUgogICAgICAgIFJTRiAtLT4gVG9rZW5pemVyCiAgICAgICAgVG9rZW5pemVyIC0tPiBPdXRwdXQKICAgIGVuZA==)
---

### Key Architectural Pillars

#### 1. Core Primitives and Memory
The system relies on a custom `Tensor` system and a suite of specialized allocators (Arena, Slab, Buddy) to manage memory without the overhead of a general-purpose heap.

#### 2. RSF Processing Pipeline
The `LayerCore` is the fundamental unit of computation. It consists of exactly four learnable tensors: `s_weight`, `t_weight`, `s_bias`, and `t_bias`. Fractal mixing is handled by the `OFTB` block, which implements a butterfly-style Haar-wavelet transform.

#### 3. Tokenization and Retrieval
JAIDE uses the **Morpheme-Guided Tokenizer (MGT)** for text decomposition and the **Structured Sequence Index (SSI)** for efficient similarity searches and knowledge retrieval.

#### 4. NSIR (Quantum-Relational Graph)
The **Non-linear Self-Similar Information Retrieval (NSIR)** system provides a hierarchical reasoning layer. It integrates quantum logic gates (Hadamard, CNOT) with classical activations to model complex relationships within a self-similar graph structure.

#### 5. Hardware Acceleration
Computation is accelerated via **Futhark** GPU kernels (CUDA/OpenCL) for the RSF flow and **Clash** for RTL hardware synthesis.

**Diagram: Hardware and Kernel Mapping**
![Hardware and Kernel Mapping](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJaaWcgTG9naWMiCiAgICAgICAgUlNGX1pbInJzZi56aWciXQogICAgICAgIEFDQ0VMWyJhY2NlbF9pbnRlcmZhY2UuemlnIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJHUFUgS2VybmVscyAoRnV0aGFyaykiCiAgICAgICAgS19GV0RbInJzZl9mbG93IChmdXRoYXJrX2tlcm5lbHMuZnV0KSJdCiAgICAgICAgS19CV0RbInJzZl9iYWNrd2FyZF9mbG93IChmdXRoYXJrX2tlcm5lbHMuZnV0KSJdCiAgICAgICAgS19TQ1RbInJzZl9zY2F0dGVyIChmdXRoYXJrX2tlcm5lbHMuZnV0KSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiSGFyZHdhcmUgVGFyZ2V0cyIKICAgICAgICBDVURBWyJOVklESUEgQjIwMCAoQ1VEQSkiXQogICAgICAgIEZQR0FbIlN5bnRoZXNpemVkIFJUTCAoQ2xhc2gpIl0KICAgIGVuZAoKICAgIFJTRl9aIC0tPiBBQ0NFTAogICAgQUNDRUwgLS0+IEtfRldECiAgICBBQ0NFTCAtLT4gS19CV0QKICAgIEtfRldEIC0tPiBDVURBCiAgICBLX1NDVCAtLT4gRlBHQQ==)
---

### Child Pages

For a deeper dive into specific components of JAIDE v40, refer to the following sections:

    Explains the Zig build system, how to enable the `gpu` flag, and the various executable targets like `jaide-inference-server` and `jaide-gpu`.
    A conceptual deep dive into the mathematical foundations of Reversible Scatter Flow and the data-flow between the neural core and the quantum-relational graph.


---

*[Back to Table of Contents](#table-of-contents) | Page 1 of 34 | Next: Getting Started — Build, Configuration & Entrypoints*

<a id="page-2"></a>

# Getting Started — Build, Configuration & Entrypoints




This page details the build infrastructure, configuration options, and primary entrypoints for the JAIDE v40 system. JAIDE utilizes the Zig build system integrated with Futhark-generated C kernels for high-performance neural processing.

## Build System & Toolchain

JAIDE is built using **Zig 0.13.0**. The build process manages both the Zig source code and the compilation of hardware-accelerated kernels.

### The Zig Build Process
The `build.zig` script defines the compilation pipeline for all system components. A critical part of the build is the integration of `futhark_kernels.c`, which contains the generated C code from Futhark for GPU and SIMD acceleration.

| Artifact | Source File | Description |
| :--- | :--- | :--- |
| `jaide` | `src/main.zig` | The primary CLI for interactive use, training, and REPL. |
| `jaide-inference-server` | `src/inference_server_main.zig` | HTTP/1.1 server for model deployment. |
| `jaide-distributed` | `src/main_distributed.zig` | Multi-node training harness (requires `-Dgpu=true`). |
| `jaide-gpu` | `src/main_gpu.zig` | Optimized single-node H100/A100 training entrypoint. |

### Build Configuration Flags
The build system supports conditional compilation through build options:
*   **GPU Acceleration**: Controlled by the `-Dgpu` flag. When enabled, it sets the `gpu_acceleration` build option to `true`, enabling distributed training and GPU-specific executables.
*   **Optimization Levels**: Standard Zig optimization levels (`Debug`, `ReleaseSafe`, `ReleaseFast`, `ReleaseSmall`) are supported via `-Doptimize`.

### Compilation Flow
The following diagram illustrates how the Zig build system orchestrates the compilation of Zig source and Futhark C kernels.

**Figure 1: JAIDE Build Pipeline**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJTb3VyY2UgU3BhY2UiCiAgICAgICAgWklHX1NSQ1sic3JjLyouemlnIl0KICAgICAgICBGVVRIQVJLX0NbInNyYy9ody9hY2NlbC9mdXRoYXJrX2tlcm5lbHMuYyJdCiAgICAgICAgRlVUSEFSS19IWyJzcmMvaHcvYWNjZWwvZnV0aGFya19rZXJuZWxzLmgiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlppZyBCdWlsZCBTeXN0ZW0gKGJ1aWxkLnppZykiCiAgICAgICAgQl9PUFRTWyJhZGRPcHRpb25zIChncHVfYWNjZWxlcmF0aW9uKSJdCiAgICAgICAgRVhFX0RFRlsiYWRkRXhlY3V0YWJsZSJdCiAgICAgICAgQ19MSU5LWyJhZGRDU291cmNlRmlsZSAoZnV0aGFya19rZXJuZWxzLmMpIl0KICAgICAgICBJTkNfUEFUSFsiYWRkSW5jbHVkZVBhdGgiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIk91dHB1dCBBcnRpZmFjdHMiCiAgICAgICAgSkFJREVfQklOWyJqYWlkZSJdCiAgICAgICAgSU5GX1NSVlsiamFpZGUtaW5mZXJlbmNlLXNlcnZlciJdCiAgICAgICAgR1BVX0JJTlsiamFpZGUtZ3B1IChDb25kaXRpb25hbCkiXQogICAgZW5kCgogICAgWklHX1NSQyAtLT4gRVhFX0RFRgogICAgRlVUSEFSS19DIC0tPiBDX0xJTksKICAgIEZVVEhBUktfSCAtLT4gSU5DX1BBVEgKICAgIEJfT1BUUyAtLT4gRVhFX0RFRgogICAgQ19MSU5LIC0tPiBFWEVfREVGCiAgICBJTkNfUEFUSCAtLT4gRVhFX0RFRgogICAgRVhFX0RFRiAtLT4gSkFJREVfQklOCiAgICBFWEVfREVGIC0tPiBJTkZfU1JWCiAgICBFWEVfREVGIC0tPiBHUFVfQklO)
---

## System Entrypoints

JAIDE provides several specialized entrypoints depending on the desired operation (inference, local training, or distributed GPU training).

### 1. Main Executable (`jaide`)
The primary entrypoint is `src/main.zig`. It handles system initialization, including the `RSF` (Reversible Scatter Flow) processor, `MGT` tokenizer, and `SSI` index. It uses a `MainConfig` struct to define default hyperparameters such as `DEFAULT_EMBEDDING_DIM` (128) and `DEFAULT_RSF_LAYERS` (4).

### 2. Inference Server (`jaide-inference-server`)
Defined in `src/inference_server_main.zig`, this entrypoint initializes an `InferenceServer` with a `ServerConfig`. It parses CLI arguments to configure the network environment:
*   `--port`: Listening port (default 8080).
*   `--host`: Bind address (default 0.0.0.0).
*   `--model`: Path to the `.jaide` model file.

### 3. GPU Training (`jaide-gpu`)
The `src/main_gpu.zig` entrypoint is designed for high-performance training on NVIDIA hardware (e.g., H100). It initializes the `GPUCoordinator` and `DistributedTrainerFuthark`, utilizing NCCL for communication primitives.

**Figure 2: Entrypoint to Core Entity Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJFbnRyeXBvaW50cyAoQ0xJKSIKICAgICAgICBNQUlOX0VYRVsiamFpZGUiXQogICAgICAgIFNSVl9FWEVbImphaWRlLWluZmVyZW5jZS1zZXJ2ZXIiXQogICAgICAgIEdQVV9FWEVbImphaWRlLWdwdSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdGllcyAoSW1wbGVtZW50YXRpb24pIgogICAgICAgIFJTRlsicHJvY2Vzc29yL3JzZi56aWc6IFJTRiJdCiAgICAgICAgTUdUWyJ0b2tlbml6ZXIvbWd0LnppZzogTUdUIl0KICAgICAgICBJTkZfU1JWWyJhcGkvaW5mZXJlbmNlX3NlcnZlci56aWc6IEluZmVyZW5jZVNlcnZlciJdCiAgICAgICAgR1BVX0NPT1JEWyJkaXN0cmlidXRlZC9ncHVfY29vcmRpbmF0b3IuemlnOiBHUFVDb29yZGluYXRvciJdCiAgICAgICAgRlVUSEFSS19UUkFJTlsiZGlzdHJpYnV0ZWQvZGlzdHJpYnV0ZWRfdHJhaW5lcl9mdXRoYXJrLnppZzogRGlzdHJpYnV0ZWRUcmFpbmVyRnV0aGFyayJdCiAgICBlbmQKCiAgICBNQUlOX0VYRSAtLT4gUlNGCiAgICBNQUlOX0VYRSAtLT4gTUdUCiAgICBTUlZfRVhFIC0tPiBJTkZfU1JWCiAgICBHUFVfRVhFIC0tPiBHUFVfQ09PUkQKICAgIEdQVV9FWEUgLS0+IEZVVEhBUktfVFJBSU4KICAgIElORl9TUlYgLS0+IFJTRg==)
---

## Configuration & Hyperparameters

The system behavior is governed by `MainConfig` and `Config` structures in `src/main.zig`.

### Default Hyperparameters
| Parameter | Default Value | Range |
| :--- | :--- | :--- |
| `embedding_dim` | 128 | 8 - 16,384 |
| `rsf_layers` | 4 | 1 - 256 |
| `batch_size` | 16 | 1 - 4,096 |
| `learning_rate` | 0.001 | 1e-10 - 10.0 |
| `sequence_length` | 64 | N/A |

### File Magic Numbers
JAIDE uses specific magic numbers for binary serialization to ensure file integrity:
*   **RSF Model**: `0x4A524653`
*   **MGT Tokenizer**: `0x4A4D4754`
*   **Ranker**: `0x4A524E4B`

---

## Testing Targets

The build system defines specific test suites that can be invoked via the Zig CLI.

| Command | Target Source | Description |
| :--- | :--- | :--- |
| `zig build test` | `src/main.zig` | Runs all unit tests across the codebase. |
| `zig build test-tensor` | `src/core/tensor.zig` | Tests tensor math, SIMD, and memory layouts. |
| `zig build test-memory` | `src/core/memory.zig` | Validates custom allocators (Arena, Slab, etc.). |

All tests are linked against `libC` and the Futhark kernels to ensure that hardware-accelerated paths are verified during the test cycle.


---

*[Back to Table of Contents](#table-of-contents) | Page 2 of 34 | Next: Architecture Overview — RSF, NSIR, and the Processing Pipeline*

<a id="page-3"></a>

# Architecture Overview — RSF, NSIR, and the Processing Pipeline




This page provides a conceptual and technical map of the JAIDE v40 architecture. It describes how the **Reversible Scatter Flow (RSF)** neural core, the **Non-linear Self-Similar Information Retrieval (NSIR)** quantum-relational graph, and the supporting processing pipeline (tokenizer, optimizer, and inference server) integrate to form a unified system.

## System Integration Map

The JAIDE v40 pipeline transitions data from "Natural Language Space" into a high-dimensional "Code Entity Space" where reasoning occurs via quantum-relational dynamics.

### Data Flow Overview

1.  **Ingestion**: Raw text is processed by the `MGT` (Morpheme-Guided Tokenizer).
2.  **Projection**: Tokens are converted into `Tensor` primitives.
3.  **Neural Core**: The `RSFLayer` performs bijective transformations using the `LayerCore` primitive.
4.  **Relational Mapping**: Embeddings are indexed in the `SSI` (Structured Sequence Index) and mapped to nodes in the `NSIR` graph.
5.  **Reasoning**: The `ReasoningOrchestrator` minimizes graph energy across `ThoughtLevel` hierarchies.
6.  **Inference**: The `InferenceServer` exposes these capabilities via a REST API.

### Component Relationship Diagram

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgIElucHV0WyJSYXcgVGV4dCBJbnB1dCJdCiAgICAgICAgTUdUWyJNR1QgKE1vcnBoZW1lLUd1aWRlZCBUb2tlbml6ZXIpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJOZXVyYWwgUHJvY2Vzc2luZyBDb3JlIChSU0YpIgogICAgICAgIFJTRl9QWyJSU0YgUHJvY2Vzc2luZyBQaXBlbGluZSJdCiAgICAgICAgT0ZUQlsiT0ZUQiAoT3J0aG9nb25hbCBGcmFjdGFsIFRyYW5zZm9ybSBCbG9jaykiXQogICAgICAgIExDWyJMYXllckNvcmUgKEFmZmluZSBDb3VwbGluZykiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlF1YW50dW0tUmVsYXRpb25hbCBTcGFjZSAoTlNJUikiCiAgICAgICAgTlNJUlsiTlNJUiBDb3JlIEdyYXBoIl0KICAgICAgICBST1siUmVhc29uaW5nIE9yY2hlc3RyYXRvciJdCiAgICAgICAgU1NJWyJTU0kgKFN0cnVjdHVyZWQgU2VxdWVuY2UgSW5kZXgpIl0KICAgIGVuZAoKICAgIElucHV0IC0tPiBNR1QKICAgIE1HVCAtLT4gUlNGX1AKICAgIFJTRl9QIC0tPiBMQwogICAgTEMgPC0tPiBPRlRCCiAgICBSU0ZfUCAtLT4gU1NJCiAgICBTU0kgPC0tPiBOU0lSCiAgICBOU0lSIDwtLT4gUk8KICAgIFJPIC0tPiBPdXRwdXRbIkluZmVyZW5jZSBSZXNwb25zZSJdCgogICAgJSUgQ29kZSBFbnRpdHkgTWFwcGluZwogICAgTUdUX0NvZGVbInNyYy90b2tlbml6ZXIvbWd0LnppZyJdIC0uLT4gTUdUCiAgICBSU0ZfQ29kZVsic3JjL3Byb2Nlc3Nvci9yc2YuemlnIl0gLS4tPiBMQwogICAgTlNJUl9Db2RlWyJzcmMvY29yZV9yZWxhdGlvbmFsL25zaXJfY29yZS56aWciXSAtLi0+IE5TSVI=)
---

## The RSF Neural Core

The **Reversible Scatter Flow (RSF)** is the fundamental computational paradigm of JAIDE. Unlike Transformers, it relies on bijective coupling layers, enabling O(1) memory backpropagation.

### LayerCore Primitive
The `LayerCore` is the only trainable primitive in the network. It consists of four tensors:
*   `s_weight` / `s_bias`: Scale parameters.
*   `t_weight` / `t_bias`: Translation parameters.

### Bijective Pipeline
The forward pass (`forwardInPlace`) and inverse pass (`inverseInPlace`) are exact algebraic inverses, ensuring no information collapse during processing.

| Operation | Logic | Memory Complexity |
| :--- | :--- | :--- |
| **Forward** | $y_1 = x_1 \odot \exp(W_s x_2 + b_s)$ | $O(1)$ (In-place) |
| **Inverse** | $x_1 = y_1 / \exp(W_s x_2 + b_s)$ | $O(1)$ (In-place) |
| **Backprop** | `backwardFromOutputs` reconstructs $x$ from $y$ | $O(1)$ (No activation cache) |

---

## NSIR Quantum-Relational Graph

The **Non-linear Self-Similar Information Retrieval (NSIR)** system handles high-level reasoning by representing knowledge as a graph of quantum states.

### Node and Edge Dynamics
Nodes in the `NSIR` graph contain a `Qubit` state representing probabilistic truth or activation. Edges have a `quality` (e.g., `entangled`, `collapsed`, `fractal`) that determines how information flows between concepts.

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJOU0lSIExvZ2ljIChzcmMvY29yZV9yZWxhdGlvbmFsL25zaXJfY29yZS56aWcpIgogICAgICAgIE5vZGVBWyJOb2RlIChRdWJpdCBTdGF0ZSkiXQogICAgICAgIE5vZGVCWyJOb2RlIChRdWJpdCBTdGF0ZSkiXQogICAgICAgIEVkZ2VbIkVkZ2UgKFF1YW50dW0gQ29ycmVsYXRpb24pIl0KICAgICAgICAKICAgICAgICBOb2RlQSAtLSAiZW50YW5nbGVOb2RlcygpIiAtLT4gRWRnZQogICAgICAgIEVkZ2UgLS0gIm1lYXN1cmUoKSIgLS0+IE5vZGVCCiAgICBlbmQKCiAgICBzdWJncmFwaCAiUmVhc29uaW5nIChzcmMvY29yZV9yZWxhdGlvbmFsL3JlYXNvbmluZy56aWcpIgogICAgICAgIFJPWyJSZWFzb25pbmdPcmNoZXN0cmF0b3IiXQogICAgICAgIFJPIC0tICJwZXJ0dXJiTG9jYWxOb2RlcygpIiAtLT4gTm9kZUEKICAgICAgICBSTyAtLSAibWluaW1pemVFbmVyZ3koKSIgLS0+IEVkZ2UKICAgIGVuZA==)
---

## The Processing Pipeline

The system operates as a continuous flow from raw input to structured reasoning.

### 1. Tokenization (MGT)
The `MGT` struct decomposes text into morphemes using a three-tier approach:
1.  **Special Tokens**: [PAD], [UNK], [BOS], [EOS].
2.  **Morphological Decomposition**: Prefixes and suffixes are prioritized.
3.  **BPE Fallback**: Byte-Pair Encoding for unknown sequences.

### 2. Structured Indexing (SSI)
The `SSI` (Structured Sequence Index) acts as the bridge between the neural embeddings and the relational graph. It uses Hamming-distance similarity to retrieve top-K candidates for reasoning.

### 3. Inference Server
The `InferenceServer` manages the lifecycle of a request:
*   **Request**: Receives JSON via `POST /v1/inference`.
*   **Execution**: Orchestrates `MGT` -> `RSF` -> `SSI` -> `NSIR`.
*   **Memory**: Uses an `ArenaAllocator` per request for high-performance, leak-free operation.

### Pipeline Data Flow Diagram

![Diagram](https://mermaid.ink/img/c2VxdWVuY2VEaWFncmFtCiAgICBwYXJ0aWNpcGFudCBVIGFzIFVzZXIgKEhUVFApCiAgICBwYXJ0aWNpcGFudCBTIGFzIEluZmVyZW5jZVNlcnZlcgogICAgcGFydGljaXBhbnQgVCBhcyBNR1QgVG9rZW5pemVyCiAgICBwYXJ0aWNpcGFudCBSIGFzIFJTRiBQcm9jZXNzb3IKICAgIHBhcnRpY2lwYW50IEcgYXMgTlNJUiBHcmFwaAoKICAgIFUtPj5TOiBQT1NUIC92MS9pbmZlcmVuY2UKICAgIFMtPj5UOiB0b2tlbml6ZShpbnB1dF90ZXh0KQogICAgVC0tPj5TOiBUb2tlbiBJRHMKICAgIFMtPj5SOiBmb3J3YXJkSW5QbGFjZShUZW5zb3IpCiAgICBSLS0+PlM6IEVtYmVkZGluZ3MKICAgIFMtPj5HOiByZWFzb24oRW1iZWRkaW5ncykKICAgIEctLT4+UzogQ29sbGFwc2VkIFN0YXRlIChSZWFzb25pbmcpCiAgICBTLT4+VTogSW5mZXJlbmNlUmVzcG9uc2UgKEpTT04p)

---

*[Back to Table of Contents](#table-of-contents) | Page 3 of 34 | Next: Core Primitives*

<a id="page-4"></a>

# Core Primitives




The core primitives represent the lowest level of the JAIDE v40 stack, providing the essential building blocks for numerical computation, memory safety, and persistent storage. These utilities are designed for high performance and strict security, serving as the foundation for the RSF neural engine and the NSIR graph system.

### Data Flow and Entity Mapping

The following diagrams illustrate how high-level concepts in the "Natural Language Space" map to specific "Code Entities" within the core primitives.

**Tensor and Numerical Space Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgICJOLURpbWVuc2lvbmFsIEFycmF5IgogICAgICAgICJSZWZlcmVuY2UgQ291bnRpbmciCiAgICAgICAgIkZpeGVkLVBvaW50IEFyaXRobWV0aWMiCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdHkgU3BhY2UgKHNyYy9jb3JlLykiCiAgICAgICAgIk4tRGltZW5zaW9uYWwgQXJyYXkiIC0tPiAiVGVuc29yW3RlbnNvci56aWc6MTU3LTE3OF0iCiAgICAgICAgIlJlZmVyZW5jZSBDb3VudGluZyIgLS0+ICJyZWZjb3VudFt0ZW5zb3IuemlnOjE2Ml0iCiAgICAgICAgIkZpeGVkLVBvaW50IEFyaXRobWV0aWMiIC0tPiAiRml4ZWQzMl8zMlt0eXBlcy56aWc6MTQ1LTE1OF0iCiAgICBlbmQKCiAgICAiVGVuc29yW3RlbnNvci56aWc6MTU3LTE3OF0iIC0tPiAiU2hhcGVbdGVuc29yLnppZzo0OS03OF0iCiAgICAiVGVuc29yW3RlbnNvci56aWc6MTU3LTE3OF0iIC0tPiAiVGVuc29ySXRlcmF0b3JbdGVuc29yLnppZzoxNC0yN10i)
**Memory and I/O Infrastructure Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgICJNZW1vcnkgQXJlbmEiCiAgICAgICAgIkZpbGUgTWFwcGluZyIKICAgICAgICAiU2VjdXJlIEVyYXNlIgogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGUgRW50aXR5IFNwYWNlIChzcmMvY29yZS8pIgogICAgICAgICJNZW1vcnkgQXJlbmEiIC0tPiAiQXJlbmFBbGxvY2F0b3JbbWVtb3J5LnppZzoxNTYtMTczXSIKICAgICAgICAiRmlsZSBNYXBwaW5nIiAtLT4gIk1NQVBbaW8uemlnOjc0LTEzOV0iCiAgICAgICAgIlNlY3VyZSBFcmFzZSIgLS0+ICJzZWN1cmVaZXJvTWVtb3J5W21lbW9yeS56aWc6MTA0Mi0xMDQ4XSIKICAgIGVuZAoKICAgICJNTUFQW2lvLnppZzo3NC0xMzldIiAtLT4gIklvQ29uZmlnW2lvLnppZzo4LTE3XSIKICAgICJBcmVuYUFsbG9jYXRvclttZW1vcnkuemlnOjE1Ni0xNzNdIiAtLT4gIk1lbW9yeUNvbmZpZ1ttZW1vcnkuemlnOjIwLTIzXSI=)
---

## [Tensor System](#2.1)

The `Tensor` struct is the primary vehicle for numerical data in JAIDE. It supports multi-dimensional shapes (up to 8 dimensions) and uses a copy-on-write (COW) mechanism to minimize memory overhead during transformations.

*   **Shape and Strides:** The `Shape` utility manages dimension sizes and calculates strides for non-contiguous memory access.
*   **Memory Integration:** Tensors can be initialized using various specialized allocators, including `ArenaAllocator`, `PoolAllocator`, and `SlabAllocator`.
*   **Iteration:** The `TensorIterator` provides a unified way to traverse tensors regardless of their underlying memory layout or striding.

For details, see [Tensor System](#2.1).

---

## [Memory Management](#2.2)

JAIDE utilizes a suite of custom allocators designed to eliminate fragmentation and provide deterministic performance for different workload patterns.

| Allocator | Purpose | File Reference |
| :--- | :--- | :--- |
| `Arena` | Fast, linear allocation for request-scoped data. | |
| `Slab` | Efficient management of fixed-size objects. | |
| `Pool` | Thread-safe allocation for uniform object types. | |
| `Buddy` | Power-of-two allocation to reduce external fragmentation. | |

The system also includes `secureZeroMemory` to ensure that sensitive data (such as model weights or decrypted tensors) is wiped from RAM immediately after use.

For details, see [Memory Management](#2.2).

---

## [I/O and Model Persistence](#2.3)

The I/O layer provides high-performance file access and a robust serialization framework for model weights and graph states.

*   **Memory Mapping:** The `MMAP` implementation allows the system to treat large files on disk as byte buffers in memory, supporting both shared and private mappings.
*   **Secure I/O:** `IoConfig` defines strict limits on file sizes and path lengths to prevent resource exhaustion attacks.
*   **Persistence:** The serialization layer handles the export and import of complex structures like the RSF `LayerCore` and NSIR `SelfSimilarRelationalGraph`.

For details, see [I/O and Model Persistence](#2.3).

---

## Shared Types

The `types.zig` module defines the primitive numerical types used throughout the engine, specifically focusing on fixed-point arithmetic to ensure cross-platform bit-determinism.

*   **Fixed-Point Arithmetic:** Types like `FixedPoint16`, `FixedPoint32`, and `Fixed32_32` provide methods for overflow-checked addition, subtraction, multiplication, and division.
*   **Error Handling:** A centralized `Error` enum defines the standard error codes used by the core primitives.


---

*[Back to Table of Contents](#table-of-contents) | Page 4 of 34 | Next: Tensor System*

<a id="page-5"></a>

# Tensor System




The **Tensor System** is the foundational data structure for all numerical computations in JAIDE. It provides a multi-dimensional array abstraction with support for arbitrary strides, copy-on-write (CoW) memory management, and SIMD-accelerated linear algebra. The system is designed for high-performance neural processing, supporting both contiguous and non-contiguous memory layouts through a robust iterator pattern.

### 1. Core Data Structures

The system revolves around the `Tensor` struct and its internal `Shape` metadata.

#### 1.1 The Tensor Struct
The `Tensor` struct manages the lifecycle of numerical data. It utilizes an atomic reference counter to support efficient sharing and a `cow` (copy-on-write) flag to trigger data duplication only when a shared tensor is modified.

| Field | Type | Description |
| :--- | :--- | :--- |
| `data` | `[]align(32) f32` | View into the active data segment. |
| `base_data` | `[]align(32) f32` | Original allocated memory block. |
| `shape` | `Shape` | Metadata describing dimensions and strides. |
| `refcount` | `*usize` | Atomic reference counter for memory management. |
| `cow` | `*bool` | Flag indicating if the tensor is shared and requires a copy before write. |

#### 1.2 Shape and Stride Calculus
The `Shape` struct handles the transformation of multi-dimensional indices into linear memory offsets. It supports up to 8 dimensions. Strides are calculated during initialization to allow for "views" (e.g., slices or transposes) without copying data.

- **Contiguous Check**: A shape is contiguous if the stride of each dimension equals the product of all subsequent dimensions.
- **Broadcasting**: The `broadcastCompatible` function determines if a tensor can be expanded to match a target shape for element-wise operations.

---

### 2. Memory Management and CoW

The Tensor system implements a **Copy-on-Write** mechanism to minimize unnecessary allocations.

1.  **Initialization**: `Tensor.init` allocates data, a `refcount` initialized to 1, and a `cow` flag set to `false`.
2.  **Retention**: `Tensor.retain` uses an atomic fetch-add (`@atomicRmw`) to increment the reference count and sets the `cow` flag to `true`.
3.  **Release**: `Tensor.release` decrements the count. If the count reaches zero, it frees the underlying `base_data`, `refcount`, and `cow` flag.
4.  **Concurrency**: The system is stress-tested for thread-safe refcounting using atomic operations.

#### Diagram: Tensor Memory Lifecycle
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIEFbIlRlbnNvci5pbml0KCkiXSAtLT4gQlsiUmVmY291bnQgPSAxLCBDT1cgPSBmYWxzZSJdCiAgICBCIC0tPiBDWyJUZW5zb3IucmV0YWluKCkiXQogICAgQyAtLT4gRFsiUmVmY291bnQgPiAxLCBDT1cgPSB0cnVlIl0KICAgIEQgLS0+IEVbIlRlbnNvci5yZWxlYXNlKCkiXQogICAgRSAtLT4gRnsiUmVmY291bnQgPT0gMD8ifQogICAgRiAtLSAiTm8iIC0tPiBHWyJLZWVwIERhdGEiXQogICAgRiAtLSAiWWVzIiAtLT4gSFsiRnJlZSBiYXNlX2RhdGEgJiBtZXRhZGF0YSJdCiAgICBEIC0tICJXcml0ZSBPcCIgLS0+IElbIkNoZWNrIENPVyBmbGFnIl0KICAgIEkgLS0gInRydWUiIC0tPiBKWyJEdXBsaWNhdGUgRGF0YSAoTmV3IFJlZmNvdW50PTEpIl0KICAgIEkgLS0gImZhbHNlIiAtLT4gS1siSW4tcGxhY2UgTW9kaWZpY2F0aW9uIl0=)
---

### 3. TensorIterator and Layouts

For non-contiguous tensors (e.g., after a transpose or slice), the `TensorIterator` provides a standard way to traverse elements in logical order regardless of physical memory layout.

- **State**: Tracks `indices` for each axis and the current linear `offset`.
- **Advance Logic**: The `advance()` method increments indices from the innermost dimension outward, updating the `offset` using pre-calculated strides.

---

### 4. Hardware Acceleration and Arithmetic

JAIDE utilizes SIMD (Single Instruction, Multiple Data) and multi-threading for tensor operations.

#### 4.1 SIMD Acceleration
The system defines a `vector_width` of 8 for `f32` operations, utilizing `@Vector(8, f32)` for parallel arithmetic. This is applied to element-wise operations like addition, subtraction, and scaling.

#### 4.2 Matrix Multiplication (Matmul)
The system provides two primary matmul implementations:
1.  **Comptime Matmul**: A specialized implementation using Zig's `comptime` for fixed-size matrices (M, K, N), enabling loop unrolling and aggressive optimization.
2.  **Multi-threaded Matmul**: For large tensors, the system partitions the workload across available CPU cores.

#### 4.3 Linear Algebra
Beyond basic arithmetic, the system supports:
- **Determinant and Inverse**: Essential for RSF (Reversible Scatter Flow) layers.
- **Fixed-Point Support**: For specialized targets, the system includes `Fixed32_32` and `FixedPoint16/32/64` types with overflow-checked arithmetic.

#### Diagram: Arithmetic Execution Path
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJMb2dpYyBTcGFjZSIKICAgICAgICBBWyJUZW5zb3IgT3AgKEFkZC9NdWwpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJDb2RlIEVudGl0eSBTcGFjZSIKICAgICAgICBBIC0tPiBCWyJUZW5zb3IuYWRkKCkiXQogICAgICAgIEIgLS0+IEN7ImlzQ29udGlndW91cz8ifQogICAgICAgIEMgLS0gIlllcyIgLS0+IERbIlNJTUQgVmVjdG9yIFBhc3MgKFZlYzgpIl0KICAgICAgICBDIC0tICJObyIgLS0+IEVbIlRlbnNvckl0ZXJhdG9yIExvb3AiXQogICAgICAgIEQgLS0+IEZbIkBWZWN0b3IoOCwgZjMyKSJdCiAgICAgICAgRSAtLT4gR1siT2Zmc2V0LWJhc2VkIGFjY2VzcyJdCiAgICBlbmQ=)
---

### 5. Integration with Allocators

The Tensor system is allocator-agnostic but provides convenience wrappers for JAIDE's custom memory management subsystems:
- **Arena**: `initWithArena` for request-scoped tensors.
- **Pool/Slab**: `initWithPool` and `initWithSlab` for fixed-size neural weights.
- **Buddy**: `initWithBuddy` for dynamic allocations with power-of-two requirements.


---

*[Back to Table of Contents](#table-of-contents) | Page 5 of 34 | Next: Memory Management*

<a id="page-6"></a>

# Memory Management




The JAIDE v40 memory management system provides a comprehensive suite of custom allocators and synchronization primitives designed for high-performance neural processing and secure data handling. The architecture emphasizes memory locality, thread safety, and deterministic resource lifecycle management through specialized allocation strategies and lock-free structures.

### Memory Configuration and Utilities

The system defines fundamental constants and utility functions for memory alignment and arithmetic safety.

| Constant | Value / Source | Description |
| :--- | :--- | :--- |
| `PageSize` | `4096` or `16384` | System-specific virtual memory page size. |
| `CACHE_LINE_SIZE` | `128` | Target alignment for preventing false sharing. |
| `secureZeroMemory` | Function | Overwrites memory with zeros using volatile operations to prevent compiler elision. |

---

### Custom Allocators

JAIDE implements several allocation strategies to minimize fragmentation and overhead across different workloads.

#### 1. Arena and ArenaAllocator
The `Arena` is a fixed-size, thread-safe buffer for rapid allocations with a single-step deallocation. The `ArenaAllocator` extends this by managing a list of dynamic buffers, providing a standard `std.mem.Allocator` interface.

*   **Key Functions**:
    *   `Arena.init(allocator, size)`: Pre-allocates a page-aligned buffer.
    *   `Arena.alloc(size, alignment)`: Thread-safe bump allocation.
    *   `Arena.secureReset()`: Zeroes all allocated memory before resetting the offset.

#### 2. Slab and Pool Allocators
Designed for uniform object sizes to eliminate external fragmentation.
*   **SlabAllocator**: Manages "slabs" of memory divided into equal-sized slots. It uses a `free_list` to track available slots.
*   **PoolAllocator**: A higher-level wrapper that manages multiple slabs, allowing the pool to grow dynamically as demand increases.

#### 3. Buddy and Page Allocators
*   **BuddyAllocator**: Implements the binary buddy system for power-of-two sized blocks, providing a balance between flexibility and fragmentation control.
*   **PageAllocator**: A thin wrapper around system-level virtual memory calls, ensuring all allocations are page-aligned.

#### 4. TrackingAllocator
A diagnostic wrapper used to monitor memory leaks and peak usage. It wraps an underlying `Allocator` and maintains counters for total allocated bytes and active allocations.

---

### Data Flow: Allocation Request Handling

The following diagram illustrates how a generic allocation request is routed through the `ArenaAllocator` logic.

**ArenaAllocator Request Flow**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIEFbIlJlcXVlc3Q6IGFyZW5hQWxsb2MobGVuLCBhbGlnbm1lbnQpIl0gLS0+IEJ7IkNoZWNrIGxlbiA9PSAwIn0KICAgIEIgLS0gIlllcyIgLS0+IENbIlJldHVybiBlbXB0eSBzbGljZSJdCiAgICBCIC0tICJObyIgLS0+IERbIkxvY2sgTXV0ZXgiXQogICAgRCAtLT4gRVsiQ2FsY3VsYXRlIGFsaWduZWRQb3MiXQogICAgRSAtLT4gRnsiRml0cyBpbiBjdXJyZW50X2J1ZmZlcj8ifQogICAgRiAtLSAiTm8iIC0tPiBHWyJDYWxsIGVuc3VyZUJ1ZmZlcihsZW4pIl0KICAgIEcgLS0+IEhbInBhcmVudF9hbGxvY2F0b3IuYWxsb2MoKSJdCiAgICBIIC0tPiBJWyJBcHBlbmQgdG8gYnVmZmVycyBsaXN0Il0KICAgIEkgLS0+IEpbIlJlc2V0IHBvcyB0byAwIl0KICAgIEogLS0+IEtbIlJlY2FsY3VsYXRlIGFsaWduZWRQb3MiXQogICAgRiAtLSAiWWVzIiAtLT4gSwogICAgSyAtLT4gTFsiVXBkYXRlIHBvcyA9IGFsaWduZWRQb3MgKyBsZW4iXQogICAgTCAtLT4gTVsiVW5sb2NrIE11dGV4Il0KICAgIE0gLS0+IE5bIlJldHVybiBwb2ludGVyIHRvIHNsaWNlIl0=)
---

### Secure and Compressed Storage

JAIDE provides specialized storage wrappers for sensitive or large-scale data.

*   **EncryptedStorage**: Wraps a memory buffer and ensures that any data written to it is encrypted at rest in RAM. It utilizes `secureZeroMemory` during `deinit` to prevent sensitive data leakage.
*   **CompressedStorage**: Implements transparent compression for large tensors or indices. It manages an internal `Arena` to handle the variable-sized output of compression algorithms.

---

### Synchronization Primitives

The system includes both mutex-based and lock-free structures for inter-thread communication.

| Structure | Type | Implementation Detail |
| :--- | :--- | :--- |
| `ThreadSafeQueue` | Mutex-based | Uses `std.Thread.Mutex` and `CondVar` for blocking `pop()` operations. |
| `LockFreeStack` | Atomic | Uses `std.atomic.Value` with `compareAndSwap` for push/pop operations to avoid lock contention. |
| `VirtualMemory` | Wrapper | Provides `mmap`/`munmap` (or `VirtualAlloc`) abstractions for direct OS memory mapping. |

### System Entity Mapping

This diagram bridges the high-level memory management concepts to the specific code entities in `src/core/memory.zig`.

**Memory Management Entity Map**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJBbGxvY2F0aW9uIFN0cmF0ZWdpZXMiCiAgICAgICAgUzFbIkJ1bXAgQWxsb2NhdGlvbiJdIC0tLSBFMVsiQXJlbmEiXQogICAgICAgIFMyWyJGaXhlZC1zaXplIFNsb3RzIl0gLS0tIEUyWyJTbGFiQWxsb2NhdG9yIl0KICAgICAgICBTM1siQmluYXJ5IFBhcnRpdGlvbmluZyJdIC0tLSBFM1siQnVkZHlBbGxvY2F0b3IiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlN0b3JhZ2UgJiBTZWN1cml0eSIKICAgICAgICBTUzFbIlZvbGF0aWxlIFplcm9pbmciXSAtLS0gRjFbInNlY3VyZVplcm9NZW1vcnkiXQogICAgICAgIFNTMlsiRW5jcnlwdGVkIFJBTSJdIC0tLSBGMlsiRW5jcnlwdGVkU3RvcmFnZSJdCiAgICAgICAgU1MzWyJWaXJ0dWFsIE1lbW9yeSJdIC0tLSBGM1siVmlydHVhbE1lbW9yeSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29uY3VycmVuY3kiCiAgICAgICAgQzFbIkF0b21pYyBPcGVyYXRpb25zIl0gLS0tIEcxWyJMb2NrRnJlZVN0YWNrIl0KICAgICAgICBDMlsiQmxvY2tpbmcgU3luYyJdIC0tLSBHMlsiVGhyZWFkU2FmZVF1ZXVlIl0KICAgIGVuZAoKICAgIEUxIC0tICJVc2VzIiAtLT4gRjEKICAgIEUyIC0tICJDb21wb25lbnQgb2YiIC0tPiBFNFsiUG9vbEFsbG9jYXRvciJdCiAgICBFMyAtLSAiQmFja3MiIC0tPiBFNVsiTlNJUiBDb3JlIE1lbW9yeSJd)

---

*[Back to Table of Contents](#table-of-contents) | Page 6 of 34 | Next: I/O and Model Persistence*

<a id="page-7"></a>

# I/O and Model Persistence




The I/O and Model Persistence layer provides the foundational infrastructure for high-performance data access and the structured serialization of the JAIDE v40 model ecosystem. It encompasses low-level memory-mapped file operations, atomic writing primitives, and the `ModelFormat` framework which orchestrates the persistence of neural weights, graph structures, and optimizer states.

## Core I/O Infrastructure

The system utilizes a custom I/O layer designed for high-throughput model loading and thread-safe parameter updates. Central to this is the `MMAP` implementation, which provides a page-aligned interface to the operating system's virtual memory subsystem.

### Memory Mapping (MMAP)
The `MMAP` struct manages file-backed memory regions, supporting both shared and private mappings. It handles automatic file resizing during appends and ensures thread safety via a dedicated mutex.

| Feature | Implementation Detail |
| :--- | :--- |
| **Page Alignment** | Uses `mem.page_size` for buffer alignment and resizing. |
| **Atomic Sync** | Supports `msync` with `MSF.SYNC` for durable writes. |
| **Bounds Safety** | Validates offsets against `actual_size` and checks for overflows. |
| **Security** | Implements `secureZeroBytes` to clear sensitive data in memory using volatile pointers. |

### Buffered and Durable Writing
The system provides `DurableWriter` and `BufferedReader` to minimize syscall overhead. For critical configuration or metadata updates, `atomicWrite` is used to ensure file integrity by writing to a temporary file and performing an atomic rename.

**I/O Data Flow and Component Interaction**

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJIaWdoLUxldmVsIFBlcnNpc3RlbmNlIgogICAgICAgIE1GWyJNb2RlbEZvcm1hdDo6ZXhwb3J0TW9kZWwiXQogICAgICAgIE1FWyJMZWFybmVkRW1iZWRkaW5nOjpzYXZlIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJJL08gTGF5ZXIiCiAgICAgICAgRFdbIkR1cmFibGVXcml0ZXIiXQogICAgICAgIEFXWyJhdG9taWNXcml0ZSJdCiAgICAgICAgTU1QWyJNTUFQIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJPUyBJbnRlcmZhY2UiCiAgICAgICAgUE9TSVhbInN0ZC5wb3NpeCAobW1hcC9tc3luYykiXQogICAgICAgIEZTWyJzdGQuZnMgKHB3cml0ZUFsbC9yZW5hbWUpIl0KICAgIGVuZAoKICAgIE1GIC0tPiBEVwogICAgTUUgLS0+IEZTCiAgICBEVyAtLT4gQVcKICAgIEFXIC0tPiBGUwogICAgTU1QIC0tPiBQT1NJWA==)
## Model Serialization Framework

The `ModelFormat` struct acts as the primary container for serializing the complete JAIDE state, including the RSF neural processor, MGT tokenizer, and Ranker components.

### Metadata and Magic Headers
Every model file begins with a magic header `JAIDE40\x00`. Metadata is stored as a JSON-escaped string containing architectural hyperparameters like `rsf_layers`, `rsf_dim`, and `mgt_vocab_size`.

### The exportModel/importModel Pipeline
1.  **Header Generation**: Writes the magic string and current version.
2.  **Metadata Serialization**: Converts `ModelMetadata` to JSON and writes it to the stream.
3.  **Component Blocks**: Each major component (RSF, MGT, Ranker) is serialized into distinct blocks.
4.  **Checksum Verification**: A SHA-256 hash is computed across the data to ensure integrity during `importModel`.

**Model Format Layout**

| Offset | Content | Type |
| :--- | :--- | :--- |
| 0x00 | `MAGIC_HEADER` | `[8]u8` |
| 0x08 | `Version` | `u32 (LE)` |
| 0x0C | `Metadata Length` | `u64 (LE)` |
| ... | `JSON Metadata` | `[]u8` |
| ... | `Component Data` | `Binary` |
| EOF - 32 | `SHA-256 Checksum` | `[32]u8` |

## Specialized Persistence Handlers

### Learned Embeddings
The `LearnedEmbedding` struct manages its own persistence via `save` and `load` methods. It uses a specific magic header `0x4A454D42` (JEMB) and stores weights as little-endian `f32` values.

### NSIR and Optimizer State
*   **NSIR Graph**: Persists the quantum-relational graph, including node states and edge qualities.
*   **SFD Optimizer**: Saves the Stochastic Fisher Diagonal state, including momentum, velocity, and the Fisher diagonal tensors, ensuring training can resume seamlessly.

**Persistence Logic Mapping**

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJNZW1vcnkgU3BhY2UiCiAgICAgICAgUlNGX1NbIlJTRiBQcm9jZXNzb3IiXQogICAgICAgIE1HVF9TWyJNR1QgVG9rZW5pemVyIl0KICAgICAgICBTRkRfU1siU0ZEIE9wdGltaXplciBTdGF0ZSJdCiAgICAgICAgTlNJUl9TWyJOU0lSIEdyYXBoIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJTZXJpYWxpemF0aW9uIExvZ2ljIgogICAgICAgIE1fSU9bIm1vZGVsX2lvLnppZyJdCiAgICAgICAgRV9JT1sibGVhcm5lZF9lbWJlZGRpbmcuemlnIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJEaXNrIFNwYWNlIgogICAgICAgIE1PREVMX0ZbIi5qYWlkZSBNb2RlbCBGaWxlIl0KICAgICAgICBFTUJfRlsiLmplbWIgRW1iZWRkaW5nIEZpbGUiXQogICAgZW5kCgogICAgUlNGX1MgLS0gIk1vZGVsRm9ybWF0IiAtLT4gTV9JTwogICAgTUdUX1MgLS0gIk1vZGVsRm9ybWF0IiAtLT4gTV9JTwogICAgU0ZEX1MgLS0gIlNGRCBQZXJzaXN0ZW5jZSIgLS0+IE1fSU8KICAgIE5TSVJfUyAtLSAiR3JhcGggRXhwb3J0IiAtLT4gTV9JTwogICAgCiAgICBNX0lPIC0tICJBdG9taWMgV3JpdGUiIC0tPiBNT0RFTF9GCiAgICBFX0lPIC0tICJCdWZmZXJlZCBXcml0ZXIiIC0tPiBFTUJfRg==)
## Hashing and Integrity Utilities

The system employs several hashing strategies for different performance and security requirements:
*   **SHA-256**: Used for model checksums and NSIR topology hashing to ensure cryptographic integrity.
*   **Blake2b256**: Utilized in `generateRuntimeSeed` for high-entropy PRNG initialization.
*   **mixHash**: A fast 64-bit non-cryptographic hash used for internal indexing and collision reduction.


---

*[Back to Table of Contents](#table-of-contents) | Page 7 of 34 | Next: Neural Processing — RSF and OFTB*

<a id="page-8"></a>

# Neural Processing — RSF and OFTB




The neural processing engine of JAIDE v40 is built upon the principle of **Invertible Neural Networks (INNs)**. Unlike traditional feed-forward architectures, the processing pipeline is designed to be bijective, allowing for exact reconstruction of inputs from outputs and $O(1)$ memory complexity during backpropagation. This is achieved through the combination of the **Reversible Scatter Flow (RSF)** coupling layers and the **Orthogonal Fractal Transform Block (OFTB)** mixer.

### Architectural Synergy

The pipeline alternates between learnable non-linear transformations (RSF) and fixed linear mixing (OFTB). This structure ensures that information is both transformed via learned parameters and diffused across the feature dimension to prevent information bottlenecks.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJQcm9jZXNzaW5nIExheWVyIChOKSIKICAgICAgICBkaXJlY3Rpb24gVEIKICAgICAgICBJbnB1dFsiSW5wdXQgVGVuc29yICh4KSJdIC0tPiBSU0ZbIlJTRiBMYXllckNvcmUgKEJpamVjdGl2ZSBDb3VwbGluZykiXQogICAgICAgIFJTRiAtLT4gT0ZUQlsiT0ZUQiAoSGFhci1XYXZlbGV0IE1peGVyKSJdCiAgICAgICAgT0ZUQiAtLT4gT3V0cHV0WyJPdXRwdXQgVGVuc29yICh5KSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiUlNGIEludGVybmFsIEZsb3ciCiAgICAgICAgZGlyZWN0aW9uIExSCiAgICAgICAgU3BsaXRbIlNwbGl0IHggaW50byB4MSwgeDIiXSAtLT4gU2NhbGVbInMgPSBleHAoTmV0X3MoeDIpKSJdCiAgICAgICAgU3BsaXQgLS0+IFRyYW5zWyJ0ID0gTmV0X3QoeDIpIl0KICAgICAgICBTY2FsZSAtLT4gQ29tYmluZVsieTEgPSB4MSAqIHMgKyB0Il0KICAgICAgICBUcmFucyAtLT4gQ29tYmluZQogICAgICAgIENvbWJpbmUgLS0+IENvbmNhdFsieSA9IFt5MSwgeDJdIl0KICAgIGVuZAoKICAgIFJTRiAtLi0+IHwiUmVmZXJzIHRvInwgTGF5ZXJDb3JlWyJwcm9jZXNzb3IvcnNmLnppZzogTGF5ZXJDb3JlIl0KICAgIE9GVEIgLS4tPiB8IlJlZmVycyB0byJ8IE9GVEJfU3RydWN0WyJwcm9jZXNzb3Ivb2Z0Yi56aWc6IE9GVEIiXQ==)

- `src/processor/rsf.zig:133-149` (LayerCore definition)
- `src/processor/oftb.zig:5-15` (OFTB definition)

---

### RSF — Reversible Scatter Flow Processor

The **RSF** is the primary learnable component of the neural core. It utilizes a coupling architecture where the input tensor is partitioned, and one half is used to compute affine transformations (scale and translation) for the other half. This ensures that the Jacobian of the transformation is triangular, making the determinant easy to compute and the function trivial to invert.

Key features of the RSF include:
*   **In-Place Operations**: Both `forwardInPlace` and `inverseInPlace` operate directly on the tensor memory to minimize allocations.
*   **Memory Efficiency**: By recomputing activations during the backward pass (using `backwardFromOutputs`), the system avoids storing intermediate states, enabling the training of extremely deep models on limited hardware.
*   **Thread Safety**: Access to weights and gradients is managed via an `RWLock` to support concurrent inference and training.

For full technical details on coupling math and GPU acceleration, see [RSF — Reversible Scatter Flow Processor](#3.1).

| Component | Code Entity | File |
| :--- | :--- | :--- |
| **Configuration** | `RSFConfig` | |
| **Core Logic** | `LayerCore` | |
| **Accelerator** | `RSFAccelerator` | |

- `src/processor/rsf.zig:147-147` (RWLock usage)
- `src/processor/rsf.zig:179-194` (LayerCore initialization)

---

### OFTB — Orthogonal Fractal Transform Block

The **OFTB** serves as a parameter-less "mixer" layer. It implements a butterfly Haar-wavelet transform that provides global communication between features. While the RSF layers focus on learning complex non-linear mappings, the OFTB ensures that every element of the tensor can influence every other element over multiple layers.

The transformation is governed by the `FRACTAL_SCALE` constant ($1/\sqrt{2}$), which preserves the norm of the tensor during the mixing process, contributing to numerical stability in deep stacks.

*   **SIMD Vectorized**: The implementation uses `@Vector(8, f32)` to process data in 8-wide chunks, significantly improving CPU performance.
*   **Fixed Operation**: Unlike RSF, the OFTB has no weights, reducing the total parameter count of the model while maintaining high expressivity.

For details on the butterfly transform and SIMD implementation, see [OFTB — Orthogonal Fractal Transform Block](#3.2).

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJPRlRCOjpmb3J3YXJkSW5QbGFjZSIKICAgICAgICBBWyJ4MVtpXSJdIC0tICIoYSAtIGIpICogMC43MDciIC0tPiBPQVsieDFfbmV3W2ldIl0KICAgICAgICBCWyJ4MltpXSJdIC0tICIoYSArIGIpICogMC43MDciIC0tPiBPQlsieDJfbmV3W2ldIl0KICAgICAgICBBIC0tPiBPQgogICAgICAgIEIgLS0+IE9BCiAgICBlbmQKICAgIAogICAgc3ViZ3JhcGggIlZlY3Rvcml6YXRpb24iCiAgICAgICAgVlsiQFZlY3Rvcig4LCBmMzIpIl0gLS0tIFBbInByb2Nlc3Nvci9vZnRiLnppZzozMS0zOSJdCiAgICBlbmQ=)

- `src/processor/oftb.zig:6-6` (FRACTAL_SCALE)
- `src/processor/oftb.zig:31-39` (SIMD vectorization logic)
- `src/processor/oftb.zig:40-45` (Scalar fallback logic)

---

### Integration and Data Flow

The neural processing layer is typically invoked by higher-level systems (such as the Inference Server or Training Harness) that manage the sequence of RSF and OFTB blocks.

1.  **Input**: A `Tensor` is provided to the `RSF` layer.
2.  **Coupling**: `LayerCore` applies scale (`s_weight`) and translation (`t_weight`) to a subset of the tensor.
3.  **Mixing**: The resulting tensor is passed to `OFTB.forwardInPlace`, which diffuses the values across the tensor's dimensions.
4.  **Recurrence**: This process repeats for the number of layers defined in `RSFConfig.max_layers`.

- `src/processor/rsf.zig:15-21` (RSFConfig)
- `src/processor/oftb.zig:22-46` (OFTB forward pass)

---

*[Back to Table of Contents](#table-of-contents) | Page 8 of 34 | Next: RSF — Reversible Scatter Flow Processor*

<a id="page-9"></a>

# RSF — Reversible Scatter Flow Processor




The **Reversible Scatter Flow (RSF)** processor is the primary neural compute engine of the JAIDE v40 architecture. It implements a bijective coupling-based architecture that allows for exact invertibility, enabling $O(1)$ memory backpropagation by reconstructing activations from outputs. RSF is designed for high-concurrency environments and features a unified interface for both CPU SIMD and GPU Futhark acceleration.

## Architectural Design

The RSF architecture is composed of a sequence of `LayerCore` blocks. Each block performs a non-linear transformation that is mathematically guaranteed to be reversible. This is achieved through a split-coupling mechanism where the input vector is divided, transformed, and then scattered back into the latent space.

### LayerCore Coupling Math

Each `LayerCore` maintains four primary parameter tensors: `s_weight` (scale), `t_weight` (translation), and their respective biases `s_bias` and `t_bias`.

The forward transformation $y = f(x)$ follows a scale-and-translate pattern:
1.  **Scale Component ($s$):** Computed as $s = \text{clip}(\text{matmul}(x, W_s) + b_s, \text{min}, \text{max})$.
2.  **Translate Component ($t$):** Computed as $t = \text{matmul}(x, W_t) + b_t$.
3.  **Coupling:** The output is produced by $y = x \cdot \exp(s) + t$.

The inverse transformation $x = f^{-1}(y)$ is computed as:
1.  $x = (y - t) \cdot \exp(-s)$.

Because $s$ and $t$ are functions of the input $x$ in a way that preserves the Jacobian structure, the transformation is bijective.

### System Entity Map

The following diagram bridges the mathematical concepts to the specific code entities in the `rsf.zig` and `accel_interface.zig` files.

**RSF Entity Association**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJSU0YgUHJvY2Vzc29yIFtzcmMvcHJvY2Vzc29yL3JzZi56aWddIgogICAgICAgIFJTRlsiUlNGIChQcm9jZXNzb3IpIl0KICAgICAgICBMQ1siTGF5ZXJDb3JlIFtzdHJ1Y3RdIl0KICAgICAgICBMQ19JTklUWyJpbml0T3duZWQoKSJdCiAgICAgICAgRldEWyJmb3J3YXJkSW5QbGFjZSgpIl0KICAgICAgICBJTlZbImludmVyc2VJblBsYWNlKCkiXQogICAgICAgIEJXRFsiYmFja3dhcmRGcm9tT3V0cHV0cygpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJIYXJkd2FyZSBBY2NlbGVyYXRpb24gW3NyYy9ody9hY2NlbC9dIgogICAgICAgIEFDQ0VMWyJSU0ZBY2NlbGVyYXRvciBbc3RydWN0XSJdCiAgICAgICAgRl9DVFhbIkZ1dGhhcmtDb250ZXh0Il0KICAgICAgICBGX0ZXRFsiZnV0aGFya19lbnRyeV9yc2ZfZm9yd2FyZCJdCiAgICAgICAgRl9CV0RbImZ1dGhhcmtfZW50cnlfcnNmX2JhY2t3YXJkIl0KICAgIGVuZAoKICAgIFJTRiAtLT58b3duc3wgTEMKICAgIExDIC0tPnx1c2VzfCBMQ19JTklUCiAgICBGV0QgLS0+fGRpc3BhdGNoZXMgdG98IEFDQ0VMCiAgICBJTlYgLS0+fENQVSBmYWxsYmFja3wgTEMKICAgIEJXRCAtLT58TygxKSBNZW1vcnl8IExDCiAgICBBQ0NFTCAtLT58d3JhcHN8IEZfQ1RYCiAgICBBQ0NFTCAtLT58Y2FsbHN8IEZfRldECiAgICBBQ0NFTCAtLT58Y2FsbHN8IEZfQldE)
## Memory and Concurrency

### O(1) Memory Backpropagation
A key feature of the RSF is `backwardFromOutputs`. Unlike standard neural networks that must store activations for every layer to compute gradients, RSF reconstructs the input of each layer by running the inverse pass during the backward step. This reduces the memory complexity of training from $O(L \cdot N)$ to $O(N)$, where $L$ is the number of layers and $N$ is the dimension.

### Thread Safety via RWLock
Each `LayerCore` contains a `std.Thread.RwLock`.
*   **Read Lock:** Acquired during `forwardInPlace` and `inverseInPlace` to allow concurrent inference passes.
*   **Write Lock:** Acquired during gradient updates or parameter synchronization to ensure atomicity.

## Hardware Acceleration (RSFAccelerator)

The RSF system integrates with GPU hardware via the `RSFAccelerator` and Futhark-generated kernels. The accelerator manages the lifecycle of GPU memory and kernel execution.

### GPU Data Flow
The data flow between the Zig `Tensor` system and the GPU context is managed through `FutharkArray` abstractions.

**GPU Acceleration Pipeline**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJIb3N0IChaaWcpIgogICAgICAgIFRbIlRlbnNvciAoZjMyKSJdCiAgICAgICAgUElOWyJQaW5uZWRNZW1vcnkgW3NyYy9ody9hY2NlbC9hY2NlbF9pbnRlcmZhY2UuemlnOjgxXSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQWNjZWxlcmF0b3IgKEZ1dGhhcmsvQ1VEQSkiCiAgICAgICAgQ1RYWyJGdXRoYXJrQ29udGV4dCBbc3JjL2h3L2FjY2VsL2FjY2VsX2ludGVyZmFjZS56aWc6MjddIl0KICAgICAgICBBUlJbIkZ1dGhhcmtBcnJheTJERjE2IFtzcmMvaHcvYWNjZWwvYWNjZWxfaW50ZXJmYWNlLnppZzoxOTNdIl0KICAgICAgICBLRVJOWyJyc2ZfZm9yd2FyZCBrZXJuZWwiXQogICAgZW5kCgogICAgVCAtLT4gUElOCiAgICBQSU4gLS0+fGN1ZGFIb3N0QWxsb2N8IEFSUgogICAgQVJSIC0tPnxmdXRoYXJrX25ld19mMTZfMmR8IENUWAogICAgQ1RYIC0tPiBLRVJOCiAgICBLRVJOIC0tPnxSZXN1bHR8IEFSUg==)
### Key Functions
| Function | Description | Source |
| :--- | :--- | :--- |
| `init()` | Initializes Futhark context, sets device 0, and configures group/tile sizes. | |
| `forwardFromTensor()` | High-level entry point to run RSF forward pass on GPU. | |
| `futhark_entry_rsf_forward` | C-interop call to the optimized GPU kernel. | |
| `sync()` | Synchronizes the GPU command queue with the host. | |

## Serialization Format (Version 4)

RSF models are persisted using a robust binary format (Version 4) that includes CRC32 checksums for every parameter tensor to ensure data integrity.

### Serialization Structure
1.  **Header:** `SAVE_VERSION` (u32), `dim` (u64), `num_layers` (u64).
2.  **Configuration:** `clip_min` (f32), `clip_max` (f32).
3.  **Layer Data:** For each layer:
    *   `s_weight`, `t_weight`, `s_bias`, `t_bias` tensors.
    *   Each tensor is preceded by its shape and followed by a CRC32 checksum of its raw data.

## Registry and Handle System

To manage large-scale models, RSF uses a registry system. Models are identified by a `RSFHandle`, which is a type-safe wrapper around a `usize` index in the global `RSFRegistry`.

*   **Registry:** A centralized store that manages the allocation and deallocation of `RSF` instances.
*   **Handles:** Prevent raw pointer leakage and allow for safe cross-thread referencing of model instances.


---

*[Back to Table of Contents](#table-of-contents) | Page 9 of 34 | Next: OFTB — Orthogonal Fractal Transform Block*

<a id="page-10"></a>

# OFTB — Orthogonal Fractal Transform Block




The **Orthogonal Fractal Transform Block (OFTB)** is a parameter-less, bijective mixing layer within the JAIDE v40 neural architecture. It implements a butterfly Haar-wavelet transform designed to provide efficient, linear-time mixing of features across the hidden dimension without requiring learned weights. By utilizing a fixed orthogonal transformation, the OFTB ensures that the energy of the signal is preserved (isometric property) while facilitating information diffusion across the tensor.

## Architectural Role

In the context of the **Reversible Scatter Flow (RSF)** pipeline, the OFTB serves as the global mixer that follows the localized non-linear transformations of the `LayerCore` coupling layers. Because it is strictly orthogonal and parameter-less, it contributes to the model's capacity to represent complex patterns through fractal-like self-similarity without increasing the parameter count or memory footprint for gradient storage.

### Data Flow Integration

The OFTB operates directly on `Tensor` data in-place. It partitions the input dimension into two halves and applies a rotation in the feature space.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJPRlRCOjpmb3J3YXJkSW5QbGFjZSIKICAgICAgICBBWyJJbnB1dCBUZW5zb3IgKHgpIl0gLS0+IEJbIlNwbGl0IGludG8geDEgKDAuLmRpbSkgYW5kIHgyIChkaW0uLjIqZGltKSJdCiAgICAgICAgQiAtLT4gQ1siU0lNRCBCdXR0ZXJmbHkgT3BlcmF0aW9uIl0KICAgICAgICBDIC0tPiBEWyJ4MV9uZXcgPSAoeDEgLSB4MikgKiAxL+KImjIiXQogICAgICAgIEMgLS0+IEVbIngyX25ldyA9ICh4MSArIHgyKSAqIDEv4oiaMiJdCiAgICAgICAgRCAtLT4gRlsiTW9kaWZpZWQgSW4tUGxhY2UgVGVuc29yIl0KICAgICAgICBFIC0tPiBGCiAgICBlbmQ=)

-
-

---

## Mathematical Implementation

The OFTB implements a normalized Haar-style butterfly transform. The transformation is defined by the constant `FRACTAL_SCALE`, which is set to $1/\sqrt{2} \approx 0.7071067811865476$ to maintain the orthogonality of the operation.

### Forward Transform
For two vector halves $a$ and $b$:
$$a_{out} = (a - b) \cdot \text{scale}$$
$$b_{out} = (a + b) \cdot \text{scale}$$

### Backward Transform (Inverse)
To reverse the operation during backpropagation or inverse inference:
$$a_{in} = (a_{out} + b_{out}) \cdot \text{scale}$$
$$b_{in} = (b_{out} - a_{out}) \cdot \text{scale}$$

| Constant | Value | Description |
| :--- | :--- | :--- |
| `FRACTAL_SCALE` | `0.7071067811865476` | The $1/\sqrt{2}$ scaling factor ensuring unit determinant. |
| `VLEN` | `8` | SIMD vector length for `f32` operations. |

-
-
-

---

## Code Entity Mapping

The following diagram maps the logical operations of the Orthogonal Fractal Transform to the specific implementation entities in `oftb.zig`.

![Diagram](https://mermaid.ink/img/Y2xhc3NEaWFncmFtCiAgICBjbGFzcyBPRlRCIHsKICAgICAgICArdXNpemUgZGltCiAgICAgICAgK0ZSQUNUQUxfU0NBTEU6IGYzMgogICAgICAgICtpbml0KGQ6IHVzaXplKSBPRlRCCiAgICAgICAgK2ZvcndhcmRJblBsYWNlKHg6ICpUZW5zb3IpICF2b2lkCiAgICAgICAgK2JhY2t3YXJkSW5QbGFjZShncmFkOiBbXWYzMikgIXZvaWQKICAgICAgICArYmFja3dhcmRJblBsYWNlU2xpY2UoZ3JhZDogW11mMzIpICF2b2lkCiAgICB9CgogICAgY2xhc3MgU0lNRF9PcHRpbWl6YXRpb24gewogICAgICAgICtAVmVjdG9yKDgsIGYzMikgdmEKICAgICAgICArQFZlY3Rvcig4LCBmMzIpIHZiCiAgICAgICAgK0BzcGxhdChzY2FsZSkgdnNjYWxlCiAgICB9CgogICAgT0ZUQiAuLj4gU0lNRF9PcHRpbWl6YXRpb24gOiB1c2VzIGZvciBsb29wcwogICAgT0ZUQiAtLXw+IFRlbnNvciA6IG9wZXJhdGVzIG9uIGRhdGE=)

-
-
-

---

## SIMD Vectorization

The OFTB is optimized for high-throughput processing using Zig's `@Vector` type. The implementation processes 8 `f32` elements per iteration (256-bit vectors) before falling back to a scalar loop for any remaining elements.

### Vectorized Forward Pass
The forward pass uses subtraction for the first half and addition for the second half to create the "scatter" effect.

1. **Load:** `va` and `vb` are loaded from the first and second halves of the tensor slice ``.
2. **Splat:** The `FRACTAL_SCALE` is splatted across a vector `vscale` ``.
3. **Compute:**
   - `x1` update: `(va - vb) * vscale` ``.
   - `x2` update: `(va + vb) * vscale` ``.

### Vectorized Backward Pass
The backward pass (used in `backwardInPlace` and `backwardInPlaceSlice`) reverses the logic to recover the original gradients or inputs.

1. **Load:** Gradients `ga` and `gb` are loaded from the split slices ``.
2. **Compute:**
   - `g1` update: `(va + vb) * vscale` ``.
   - `g2` update: `(vb - va) * vscale` ``.

-
-

---

## Integration and Error Handling

The `OFTB` struct provides safety checks to ensure tensor dimensions are compatible with the split-mixing logic.

- **Initialization:** `init(d: usize)` requires a non-zero dimension and asserts that the dimension will not cause an overflow when doubled (as the block operates on `2 * dim` total elements) ``.
- **Validation:** Both `forwardInPlace` and `backwardInPlace` verify that the provided `Tensor` or slice contains at least `self.dim * 2` elements ``.
- **Memory:** The `OFTB` is essentially a metadata container (`dim`); `deinit` simply sets the struct to `undefined` as it owns no heap memory ``.

-
-

---

*[Back to Table of Contents](#table-of-contents) | Page 10 of 34 | Next: Tokenizer and Retrieval*

<a id="page-11"></a>

# Tokenizer and Retrieval




The **Tokenizer and Retrieval** subsystem provides the bridge between raw natural language and the high-dimensional vector spaces processed by the RSF core. It handles the transformation of text into discrete tokens, the efficient storage of these sequences in a structured index, and the ranking of candidates during inference.

## Overview

The pipeline consists of three primary components:
1.  **MGT (Morpheme-Guided Tokenizer)**: Decomposes text into a hybrid of morphological units and BPE tokens.
2.  **SSI (Structured Sequence Index)**: A high-performance hash tree for storing and retrieving sequence segments.
3.  **Ranker**: A scoring engine that evaluates sequence candidates using n-gram weights, diversity heuristics, and Jaccard similarity.

### Data Flow: Text to Candidate Retrieval

The following diagram illustrates how natural language is transformed into indexed entities within the `SSI` and subsequently scored by the `Ranker`.

**System Entity Mapping: Language to Index**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgIElucHV0WyJSYXcgVGV4dCBTdHJpbmciXQogICAgZW5kCgogICAgc3ViZ3JhcGggIk1HVCDigJQgVG9rZW5pemF0aW9uIgogICAgICAgIE1HVF9FWyJNR1QuZW5jb2RlKCkiXQogICAgICAgIE1vcnBoZW1lc1siTW9ycGhlbWUgRGVjb21wb3NpdGlvbiJdCiAgICAgICAgQlBFWyJCUEUgRmFsbGJhY2siXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlNTSSDigJQgU3RvcmFnZSAmIEluZGV4aW5nIgogICAgICAgIFNTSV9Sb290WyJTU0kuTm9kZSAoUm9vdCkiXQogICAgICAgIFNTSV9CdWNrZXRbIlNTSS5idWNrZXRJbmRleCgpIl0KICAgICAgICBTU0lfU2VnWyJTU0kuU2VnbWVudCJdCiAgICAgICAgU1NJX0NvbGxbIlNTSS5Db2xsaXNpb25Ob2RlIl0KICAgIGVuZAoKICAgIElucHV0IC0tPiBNR1RfRQogICAgTUdUX0UgLS0+IE1vcnBoZW1lcwogICAgTW9ycGhlbWVzIC0tPiBCUEUKICAgIEJQRSAtLT58InRva2VuX2lkICh1MzIpInwgU1NJX0J1Y2tldAogICAgU1NJX0J1Y2tldCAtLT4gU1NJX1Jvb3QKICAgIFNTSV9Sb290IC0tPiBTU0lfU2VnCiAgICBTU0lfU2VnIC0tPiBTU0lfQ29sbA==)
---

## MGT — Morpheme-Guided Tokenizer

The `MGT` (Morpheme-Guided Tokenizer) implements a three-tier tokenization strategy. Unlike standard BPE-only tokenizers, MGT prioritizes morphological decomposition (prefixes, roots, and suffixes) to better handle highly inflected languages and maintain semantic consistency.

*   **Three-Tier Pipeline**: It first checks for special tokens (e.g., `[BOS]`, `[EOS]`), then attempts to split words into known morphemes using `prefixes` and `suffixes` tables, and finally falls back to a trained Byte Pair Encoding (BPE) for unknown substrings.
*   **Anchor Tracking**: The tokenizer identifies "anchors"—statistically significant tokens that serve as high-confidence points for the `Ranker` and `SSI` search.
*   **Special Tokens**: Supports standard reserved IDs: `[PAD]` (0), `[UNK]` (1), `[BOS]` (2), and `[EOS]` (3).

For implementation details on the BPE training algorithm and vocabulary persistence, see [MGT — Morpheme-Guided Tokenizer](#4.1).

---

## SSI — Structured Sequence Index

The `SSI` (Structured Sequence Index) is a 64-bucket hash tree designed for O(log N) retrieval of sequence segments. It acts as the primary memory for the system's "Code Entity Space."

*   **Data Model**: Data is stored in `Segment` structs, which contain token arrays, positional metadata, and pre-calculated scores.
*   **Collision Handling**: Uses `CollisionNode` chains to handle hash collisions within the tree buckets.
*   **Similarity Search**: Supports Hamming-distance based similarity searches to find relevant context even with imperfect matches.
*   **Tensor Integration**: The index can be exported to a 134-column `Tensor` layout for bulk processing by the RSF or GPU kernels.

For details on the tree balancing and binary serialization format, see [SSI — Structured Sequence Index](#4.2).

**Code Entity Mapping: SSI Internal Structure**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJTU0kgU3RydWN0IFtzcmMvaW5kZXgvc3NpLnppZ10iCiAgICAgICAgUm9vdFsicm9vdDogKk5vZGUiXQogICAgICAgIEhlaWdodFsiaGVpZ2h0OiB1c2l6ZSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiTm9kZSBTdHJ1Y3QiCiAgICAgICAgQ2hpbGRyZW5bImNoaWxkcmVuOiBbXT8qTm9kZSJdCiAgICAgICAgU2VnUHRyWyJzZWdtZW50OiA/U2VnbWVudCJdCiAgICAgICAgQ2hhaW5bImNvbGxpc2lvbl9jaGFpbjogPypDb2xsaXNpb25Ob2RlIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJTZWdtZW50IFN0cnVjdCIKICAgICAgICBUb2tlbnNbInRva2VuczogW111MzIiXQogICAgICAgIFBvc1sicG9zaXRpb246IHU2NCJdCiAgICAgICAgU2NvcmVbInNjb3JlOiBmMzIiXQogICAgZW5kCgogICAgUm9vdCAtLT4gQ2hpbGRyZW4KICAgIENoaWxkcmVuIC0tPiBOb2RlMlsiTm9kZSAoTGVhZikiXQogICAgTm9kZTIgLS0+IFNlZ1B0cgogICAgTm9kZTIgLS0+IENoYWluCiAgICBTZWdQdHIgLS0+IFRva2VucwogICAgU2VnUHRyIC0tPiBQb3MKICAgIFNlZ1B0ciAtLT4gU2NvcmU=)
---

## Ranker — Sequence Scoring and Candidate Evaluation

The `Ranker` evaluates the relevance of retrieved segments against a query or the current context. It uses a multi-objective scoring function calibrated by `RankerConfig` constants.

*   **N-Gram Weighting**: Implements decaying weights for different n-gram lengths to prioritize longer, more specific matches.
*   **Diversity & Proximity**: Scores are adjusted based on `DIVERSITY_WEIGHT` (uniqueness of tokens) and `PROXIMITY_WEIGHT` (distance to known anchors in the `SSI`).
*   **Similarity Metrics**: Combines Jaccard similarity and token overlap to ensure retrieved candidates are semantically aligned with the input.
*   **Parallel Scoring**: Supports multi-threaded evaluation of large candidate heaps using `topKHeap` structures.

For information on weight calibration via gradient descent and streaming ranking, see [Ranker — Sequence Scoring and Candidate Evaluation](#4.3).


---

*[Back to Table of Contents](#table-of-contents) | Page 11 of 34 | Next: MGT — Morpheme-Guided Tokenizer*

<a id="page-12"></a>

# MGT — Morpheme-Guided Tokenizer




The **Morpheme-Guided Tokenizer (MGT)** is a high-performance, three-tier tokenization system designed for the JAIDE v40 architecture. Unlike standard Byte-Pair Encoding (BPE) systems, MGT prioritizes morphological integrity by decomposing words into prefixes, roots, and suffixes before falling back to subword merging. This approach ensures that the resulting tokens align more closely with semantic and grammatical structures, particularly in agglutinative languages.

### Tokenization Pipeline

The MGT processes input text through a hierarchical pipeline to convert raw strings into a sequence of integer token IDs.

1.  **Special Token Identification**: The tokenizer first scans for reserved control sequences like `.
2.  **Morphological Decomposition**: The system attempts to strip known prefixes and suffixes from words to isolate the root. This is guided by pre-defined lists of common morphemes.
3.  **BPE Fallback**: If a word or its decomposed parts are not found in the vocabulary, the tokenizer applies Byte-Pair Encoding (BPE) merges based on learned priority pairs.

#### Data Flow and Implementation

The `MGT` struct manages the vocabulary and the state required for encoding and decoding.

| Component | Code Entity | Role |
| :--- | :--- | :--- |
| **Vocabulary** | `token_to_id` | Map for fast string-to-ID lookups. |
| **Inverse Map** | `id_to_token` | Map for decoding IDs back to strings. |
| **Morpheme Stores** | `prefixes`, `suffixes`, `roots` | Specialized maps for morphological components. |
| **BPE Logic** | `bpe_pairs` | Stores merge priorities for subword tokenization. |
| **Anchors** | `anchors` | Tracks high-importance tokens used for sequence alignment. |

### Architecture and Memory Integration

MGT is designed to work seamlessly with JAIDE's custom memory management system, allowing it to be initialized within different allocation contexts (Arena, Pool, or Buddy allocators).

#### Tokenizer Initialization Flow
The following diagram illustrates how the `MGT` is initialized and how it interacts with the `core_memory` primitives.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgICJSYXdWb2NhYiJbIlJhdyBWb2NhYnVsYXJ5IChTdHJpbmdzKSJdCiAgICAgICAgIkFuY2hvckxpc3QiWyJBbmNob3IgTGlzdCAoU3RyaW5ncykiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGUgRW50aXR5IFNwYWNlIChzcmMvdG9rZW5pemVyL21ndC56aWcpIgogICAgICAgICJNR1RfaW5pdCJbIk1HVC5pbml0KCkiXQogICAgICAgICJpbml0RW1wdHkiWyJpbml0RW1wdHkoKSJdCiAgICAgICAgImluaXRNb3JwaGVtZXMiWyJpbml0TW9ycGhlbWVzKCkiXQogICAgICAgICJhZGRUb2tlbiJbImFkZFRva2VuKCkiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIk1lbW9yeSBNYW5hZ2VtZW50IChzcmMvY29yZS9tZW1vcnkuemlnKSIKICAgICAgICAiQXJlbmEiWyJBcmVuYUFsbG9jYXRvciJdCiAgICAgICAgIlBvb2wiWyJQb29sQWxsb2NhdG9yIl0KICAgICAgICAiQnVkZHkiWyJCdWRkeUFsbG9jYXRvciJdCiAgICBlbmQKCiAgICAiUmF3Vm9jYWIiIC0tPiAiTUdUX2luaXQiCiAgICAiQW5jaG9yTGlzdCIgLS0+ICJNR1RfaW5pdCIKICAgICJBcmVuYSIgLS0gImFsbG9jYXRvcigpIiAtLT4gIk1HVF9pbml0IgogICAgIk1HVF9pbml0IiAtLT4gImluaXRFbXB0eSIKICAgICJNR1RfaW5pdCIgLS0+ICJhZGRUb2tlbiIKICAgICJhZGRUb2tlbiIgLS0+ICJpbml0TW9ycGhlbWVzIg==)
### Technical Details

#### Special Token IDs
The MGT reserves specific IDs for control flow and padding, ensuring consistency across the training and inference pipelines.

*   **`.
*   **`.
*   **`.
*   **`.

#### Morphological Logic
The `initMorphemes` function populates the internal maps with common linguistic units. For example, it includes English prefixes like "un-", "re-", and "pre-", as well as Hungarian morphemes like "meg-", "szét-", and various case endings like "-ban/-ben". This hybrid approach allows the model to handle complex word forms more efficiently than standard subword tokenizers.

#### Batch Encoding and Tensor Integration
MGT is designed to output results directly into the JAIDE `Tensor` system. When encoding a batch of text, the tokenizer produces a `core_tensor.Tensor` object containing the token IDs, which can then be fed into the RSF neural core.

### BPE Training and Persistence
The BPE algorithm implemented in MGT follows a standard frequency-based merging strategy but is constrained by the morphological boundaries established during the decomposition phase.

1.  **Pair Counting**: The system identifies the most frequent adjacent pairs of tokens in the training corpus.
2.  **Merge Rule Generation**: High-frequency pairs are assigned a `BPEMerge` priority.
3.  **Vocabulary Persistence**: The resulting vocabulary, including BPE rules and morphological maps, can be serialized and deserialized using the `I/O` utilities in `src/core/io.zig`.

#### Tokenizer Data Structure Mapping
This diagram bridges the conceptual tokenizer components to their specific Zig implementations.

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJUb2tlbml6ZXIgQ29uY2VwdHMiCiAgICAgICAgIlN1YndvcmRNZXJnaW5nIlsiU3Vid29yZCBNZXJnaW5nIl0KICAgICAgICAiU3BlY2lhbFRva2VucyJbIkNvbnRyb2wgU2VxdWVuY2VzIl0KICAgICAgICAiTW9ycGhvbG9neSJbIk1vcnBoZW1lIExvZ2ljIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJaaWcgSW1wbGVtZW50YXRpb24gKHNyYy90b2tlbml6ZXIvbWd0LnppZykiCiAgICAgICAgIkJQRU1lcmdlIlsic3RydWN0IEJQRU1lcmdlIl0KICAgICAgICAiU1RfQ29uc3RzIlsiU1BFQ0lBTF9UT0tFTlMgY29uc3RhbnRzIl0KICAgICAgICAiUHJlZml4TWFwIlsicHJlZml4ZXM6IFN0cmluZ0hhc2hNYXAiXQogICAgICAgICJTdWZmaXhNYXAiWyJzdWZmaXhlczogU3RyaW5nSGFzaE1hcCJdCiAgICBlbmQKCiAgICAiU3Vid29yZE1lcmdpbmciIC0tPiAiQlBFTWVyZ2UiCiAgICAiU3BlY2lhbFRva2VucyIgLS0+ICJTVF9Db25zdHMiCiAgICAiTW9ycGhvbG9neSIgLS0+ICJQcmVmaXhNYXAiCiAgICAiTW9ycGhvbG9neSIgLS0+ICJTdWZmaXhNYXAi)

---

*[Back to Table of Contents](#table-of-contents) | Page 12 of 34 | Next: SSI — Structured Sequence Index*

<a id="page-13"></a>

# SSI — Structured Sequence Index




The Structured Sequence Index (SSI) is a high-performance, hierarchical data structure designed for indexing and retrieving token sequences. It utilizes a 64-bucket hash tree architecture to provide efficient storage and similarity search capabilities, bridging the gap between raw token streams and structured relational graphs.

## SSI Tree Architecture

The SSI is implemented as a multi-level hash tree where each internal node branches into 64 possible buckets. This structure allows for rapid narrowing of the search space based on hash prefixes.

### Data Model
The SSI relies on three primary data structures defined in `src/index/ssi.zig`:

*   **Segment**: The fundamental unit of storage, containing a sequence of tokens, its global position, a relevance score, and an anchor hash used for structural alignment.
*   **Node**: A branch or leaf in the tree. Branch nodes contain an array of 64 optional child pointers, while leaf nodes store segments.
*   **CollisionNode**: A linked list structure attached to leaf nodes to handle hash collisions, ensuring that multiple segments with identical hash prefixes can be stored without loss.

### Structural Constants
| Constant | Value | Description |
| :--- | :--- | :--- |
| `bucket_width` | 6 | Number of bits used per level (2^6 = 64 buckets). |
| `bucket_count` | 64 | Total children per internal node. |
| `tensor_width` | 134 | Columns used for Tensor export/import. |
| `max_height` | 6 | Maximum depth of the hash tree. |

## Search and Retrieval

SSI supports both exact matching and similarity-based retrieval. The retrieval process uses a priority queue to maintain the "Top K" most relevant results based on a combination of hash similarity and segment scores.

### Hamming-Distance Similarity
For similarity search, the system evaluates the distance between the search key and stored segment hashes. This is often offloaded to hardware-accelerated components like `SSISearch.hs` which implements a Mealy state machine for tree traversal.

### Hardware-Accelerated Search Flow
The following diagram illustrates the transition from a software search request to the hardware search logic.

**Diagram: SSI Search Request Flow**
![SSI Search Request Flow](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJTb2Z0d2FyZSAoc3JjL2luZGV4L3NzaS56aWcpIgogICAgICAgIEFbIlNTSS5yZXRyaWV2ZVRvcEsoKSJdIC0tPiBCWyJTU0kuYnVja2V0SW5kZXgoKSJdCiAgICAgICAgQiAtLT4gQ1siUmVjdXJzaXZlIFRyYXZlcnNhbCJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiSGFyZHdhcmUgTG9naWMgKHNyYy9ody9ydGwvU1NJU2VhcmNoLmhzKSIKICAgICAgICBDIC0uLT4gRFsiU2VhcmNoUmVxdWVzdCAoc2VhcmNoS2V5LCByb290QWRkcikiXQogICAgICAgIEQgLS0+IEV7InNzaVNlYXJjaFQgU3RhdGUifQogICAgICAgIEUgLS0gIklkbGUiIC0tPiBGWyJGZXRjaGluZyJdCiAgICAgICAgRSAtLSAiRmV0Y2hpbmciIC0tPiBHWyJDb21wYXJpbmciXQogICAgICAgIEcgLS0+IEh7ImNoZWNrTm9kZSgpIn0KICAgICAgICBIIC0tICJNYXRjaCIgLS0+IElbIlNlYXJjaFJlc3VsdCAoZm91bmQ9VHJ1ZSkiXQogICAgICAgIEggLS0gIk1pc21hdGNoIiAtLT4gSlsiTmV4dCBDaGlsZCAobGVmdENoaWxkL3JpZ2h0Q2hpbGQpIl0KICAgICAgICBKIC0tPiBGCiAgICBlbmQ=)
## Tensor Export and Layout

The SSI can export its entire state into a `Tensor` format for neural processing or persistence. This export uses a specific 134-column layout to represent the segment data and its structural metadata.

### 134-Column Layout Mapping
When a `Segment` is converted to a tensor row, the data is packed as follows:

1.  **Metadata (Columns 0-5)**: Includes `position` (split into low32/high32), `score` (bit-casted f32), and `anchor_hash`.
2.  **Token Data (Columns 6-133)**: Up to 128 tokens are stored sequentially. If a segment has fewer tokens, the remaining columns are typically padded.

### Key Functions
*   **`low32` / `high32`**: Utility functions to split 64-bit hashes/positions into 32-bit components for tensor compatibility.
*   **`joinU64`**: Reconstructs 64-bit values from two 32-bit tensor columns.
*   **`refreshHash`**: Recomputes the Merkle-style hash for a node based on its children (branch) or segments (leaf).

## Compaction and Balancing

As segments are inserted or deleted, the tree may become unbalanced or fragmented. The SSI implementation includes logic to maintain structural integrity:

1.  **Recursive Deinitialization**: Ensures that all dynamically allocated `Node` children and `CollisionNode` chains are freed correctly to prevent memory leaks.
2.  **Hash Refreshing**: Every insertion triggers a bottom-up hash update (`refreshHash`), ensuring the root hash always represents the current state of the index.
3.  **Leaf Insertion**: Logic in `insertIntoLeaf` handles the transition from an empty leaf to a populated one, including the initialization of the `Segment` data.

## Data Flow: Sequence to Index

This diagram bridges the natural language tokenization process with the SSI storage model.

**Diagram: Token Sequence Ingestion**
![Token Sequence Ingestion](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJJbnB1dCBTcGFjZSIKICAgICAgICBUb2tlbnNbInUzMiBUb2tlbiBTdHJlYW0iXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlNTSSBFbnRpdHkgU3BhY2UgKHNyYy9pbmRleC9zc2kuemlnKSIKICAgICAgICBTRUdbIlNlZ21lbnQgU3RydWN0Il0KICAgICAgICBOT0RFWyJOb2RlIFN0cnVjdCJdCiAgICAgICAgQ09MTFsiQ29sbGlzaW9uTm9kZSJdCiAgICBlbmQKCiAgICBUb2tlbnMgLS0+fCJTZWdtZW50LmluaXQoKSJ8IFNFRwogICAgU0VHIC0tPnwiU1NJLmluc2VydCgpInwgTk9ERQogICAgTk9ERSAtLSAiSGFzaCBDb2xsaXNpb24iIC0tPiBDT0xMCiAgICBOT0RFIC0tICI2LWJpdCBQcmVmaXgiIC0tPiBCVUNLRVRbIkJ1Y2tldCBbMC4uNjNdIl0KICAgIAogICAgc3ViZ3JhcGggIlNlcmlhbGl6YXRpb24iCiAgICAgICAgTk9ERSAtLT58IlNTSS5yZWZyZXNoSGFzaCgpInwgUm9vdEhhc2hbIk1lcmtsZSBSb290Il0KICAgICAgICBTRUcgLS0+fCJUZW5zb3IgRXhwb3J0InwgVFsiVGVuc29yICgxMzQgY29scykiXQogICAgZW5k)


---

*[Back to Table of Contents](#table-of-contents) | Page 13 of 34 | Next: Ranker — Sequence Scoring and Candidate Evaluation*

<a id="page-14"></a>

# Ranker — Sequence Scoring and Candidate Evaluation




The Ranker subsystem is responsible for the final evaluation and selection of sequence candidates retrieved from the **SSI (Structured Sequence Index)**. It operates as a multi-stage scoring engine that combines n-gram frequency weights, heuristic diversity measures, and similarity metrics (Jaccard/MinHash) to produce a normalized score for inference and training.

## Architecture and Core Scoring Logic

The `Ranker` struct manages the scoring state, including learned n-gram weights and parameters for Locality Sensitive Hashing (LSH). It acts as the bridge between raw token sequences and the relational importance stored in the SSI.

### Multi-Stage Scoring Pipeline

The primary entry point for evaluation is `scoreSequence`, which implements a weighted sum of several components:

1.  **N-Gram Weighting**: The system iterates through n-grams (up to `num_ngrams`) and retrieves their corresponding segment scores from the SSI. Weights are initialized with a harmonic decay ($1/n$).
2.  **Diversity Heuristic**: Measures the ratio of unique tokens to total tokens to penalize repetitive sequences.
3.  **Anchor Proximity**: Evaluates the distance between tokens and known morphological anchors within the SSI graph.
4.  **Normalization**: The raw score is clamped and normalized against `MAX_RAW_SCORE` (default 100.0).

### Candidate Evaluation Flow
This diagram illustrates how a query sequence is processed through the Ranker's internal components.

Title: Ranker Sequence Scoring Flow
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJSYW5rZXIgRXZhbHVhdGlvbiBQaXBlbGluZSIKICAgICAgICBBWyJzY29yZVNlcXVlbmNlV2l0aFF1ZXJ5Il0gLS0+IEJbInNjb3JlU2VxdWVuY2UiXQogICAgICAgIEEgLS0+IENbImNvbXB1dGVUb2tlbk92ZXJsYXAiXQogICAgICAgIEEgLS0+IERbImNvbXB1dGVKYWNjYXJkU2ltaWxhcml0eSJdCiAgICAgICAgCiAgICAgICAgQiAtLT4gQjFbIk4tR3JhbSBTY29yaW5nIl0KICAgICAgICBCIC0tPiBCMlsiY29tcHV0ZVRva2VuRGl2ZXJzaXR5Il0KICAgICAgICBCIC0tPiBCM1siYW5jaG9yUHJveGltaXR5Il0KICAgICAgICAKICAgICAgICBCMSAtLT4gU1NJWyJTU0kuZ2V0U2VnbWVudChoKSJdCiAgICAgICAgCiAgICAgICAgQyAtLT4gRVsiV2VpZ2h0ZWQgU3VtIl0KICAgICAgICBEIC0tPiBFCiAgICAgICAgQiAtLT4gRQogICAgICAgIAogICAgICAgIEUgLS0+IEZbIm1hdGguY2xhbXAoMC4wLCAxLjApIl0KICAgIGVuZA==)
## Similarity and Signature Metrics

To handle large-scale retrieval, the Ranker implements MinHash and Jaccard similarity to approximate the overlap between sequences without exhaustive comparison.

*   **Jaccard Similarity**: Implemented using an `AutoHashMap` to calculate the intersection over union of token sets.
*   **MinHash/LSH**: The Ranker generates `num_hash_functions` signatures for each sequence. These signatures are used for fast approximate similarity searches.
    *   **Signature Generation**: Uses `stableHash` with a rotating seed generated from `HASH_SEED_MULTIPLIER_A` and `B`.
    *   **LSH Signatures**: `computeMinHashSignatures` populates a slice of `u64` representing the minimum hash values seen across the token sequence.

## Hardware Acceleration: RankerCore

The Ranker is designed for high-throughput execution via a dedicated hardware component defined in Clash (Haskell-to-RTL). The `RankerCore` handles the performance-critical path of score accumulation and position bias calculation.

### RankerCore Logic
The hardware implementation utilizes a Mealy state machine (`rankerT`) to process `RankRequest` packets.

*   **Position Bias**: Implements a bias based on the token's position in the segment: $bias = \text{scale} / (\text{pos} + 1)$.
*   **State Tracking**: The `RankerState` tracks the `lastQuery` and `stateCounter` to handle sequential ranking requests for the same query hash.

Title: Hardware-Software Ranker Interface
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJTb2Z0d2FyZSAoWmlnKSIKICAgICAgICBaWyJyYW5rZXIuemlnIl0gLS0gInN0YWJsZUhhc2giIC0tPiBRWyJRdWVyeUhhc2g2NCJdCiAgICAgICAgWiAtLSAic3NpLmdldFNlZ21lbnQiIC0tPiBTWyJTZWdtZW50SUQ2NCJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiSGFyZHdhcmUgKENsYXNoIFJUTCkiCiAgICAgICAgUlsiUmFua2VyQ29yZS5ocyJdCiAgICAgICAgU1RbIlJhbmtlclN0YXRlIl0KICAgICAgICAKICAgICAgICBRIC0tPiBSCiAgICAgICAgUyAtLT4gUgogICAgICAgIAogICAgICAgIFIgLS0+IEJDWyJjb21wdXRlUG9zaXRpb25CaWFzIl0KICAgICAgICBCQyAtLT4gRlNbImNvbXB1dGVGaW5hbFNjb3JlIl0KICAgICAgICBGUyAtLT4gUlJbIlJhbmtSZXN1bHQiXQogICAgZW5kCiAgICAKICAgIFJSIC0tPiBa)
## Streaming and Calibration

The Ranker supports real-time evaluation of long sequences through a sliding window mechanism.

### Streaming Ranking
The system uses a `STREAMING_BUFFER_SIZE` (1024) and `STREAMING_WINDOW_SIZE` (512) to process incoming token streams. This allows the ranker to maintain a local context and provide scores for sequences that exceed the typical SSI segment length.

### Weight Calibration via Gradient Descent
The `ngram_weights` are not static. The `calibrateWeights` function implements a basic gradient descent step to adjust n-gram importance based on an error signal (the difference between the `target_score` and the current `predicted_score`).

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `LEARNING_RATE` | 0.01 | Step size for weight updates. |
| `DIVERSITY_WEIGHT` | 0.3 | Importance of unique token distribution. |
| `PROXIMITY_WEIGHT` | 0.3 | Importance of morphological anchor proximity. |
| `BASE_SCORE_WEIGHT` | 0.4 | Weight of the raw SSI segment score. |

## Model Persistence

Ranker state (n-gram weights and LSH parameters) is persisted to disk to maintain consistency across inference sessions.

*   **Export**: `exportToFile` writes the `num_ngrams`, `num_hash_functions`, `seed`, and the full `ngram_weights` and `lsh_hash_params` buffers.
*   **Import**: `importFromFile` restores these parameters and re-initializes the Ranker instance.


---

*[Back to Table of Contents](#table-of-contents) | Page 14 of 34 | Next: NSIR — Quantum-Relational Graph System*

<a id="page-15"></a>

# NSIR — Quantum-Relational Graph System




The **Non-linear Self-Similar Information Retrieval (NSIR)** system is the knowledge representation and reasoning backbone of JAIDE v40. Unlike traditional vector databases, NSIR represents information as a dynamic graph where nodes possess quantum states (superposition, entanglement) and edges reflect relational quality and fractal dimensionality. This architecture allows the system to perform non-linear reasoning by minimizing the global "energy" of the graph through quantum-inspired optimization.

### System Architecture Overview

The NSIR system bridges the gap between raw unstructured data and high-order reasoning by transforming extracted triplets into a self-similar graph structure.

**NSIR Knowledge Flow**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJJbmdlc3Rpb24gTGF5ZXIiCiAgICAgICAgQVsiQ1JFVlBpcGVsaW5lIl0gLS0gIlJlbGF0aW9uYWxUcmlwbGV0IiAtLT4gQlsiU2VsZlNpbWlsYXJSZWxhdGlvbmFsR3JhcGgiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlByb2Nlc3NpbmcgQ29yZSIKICAgICAgICBCIDwtLT4gQ1siUmVhc29uaW5nT3JjaGVzdHJhdG9yIl0KICAgICAgICBDIDwtLT4gRFsiRVNTTyBPcHRpbWl6ZXIiXQogICAgICAgIEMgPC0tPiBFWyJDaGFvc0NvcmVLZXJuZWwiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlF1YW50dW0gRXhlY3V0aW9uIgogICAgICAgIEQgLS0gIkpvYiBTdWJtaXNzaW9uIiAtLT4gRlsiSUJNIFF1YW50dW0gLyBpYm1fYnJpc2JhbmUiXQogICAgICAgIEQgLS0gIlNpbXVsYXRpb24iIC0tPiBHWyJSZWxhdGlvbmFsUXVhbnR1bUxvZ2ljIl0KICAgIGVuZAoKICAgIEIgLS0gImV4cG9ydE5vZGVFbWJlZGRpbmdzIiAtLT4gSFsiTmV1cmFsIFNwYWNlIChSU0YpIl0=)
---

### NSIR Core — Graph Structure and Quantum Operations
The foundational data structure is the `SelfSimilarRelationalGraph`. It manages `Node` objects, which contain `Qubit` states, and `Edge` objects defined by an `EdgeQuality` enum.

- **Quantum States**: Nodes use complex amplitudes (alpha/beta) to represent information uncertainty.
- **Edge Quality**: Relationships transition through states: `superposition`, `entangled`, `coherent`, `collapsed`, and `fractal`.
- **Topology Hashing**: The graph maintains structural integrity using SHA-256 Merkle-style hashing of its state.

For details, see [NSIR Core — Graph Structure and Quantum Operations](#5.1).

---

### Reasoning Orchestrator and Energy Minimization
The `ReasoningOrchestrator` manages the lifecycle of "thought" within the graph. It operates across a `ThoughtLevel` hierarchy: `local`, `global`, and `meta`.

- **Energy Formula**: Reasoning is framed as an optimization problem where the system seeks to minimize a "graph energy" function defined by connectivity and quantum coherence.
- **Cycle Execution**: The orchestrator coordinates the `ChaosCoreKernel` for entropy injection and the `ESSO` (Entangled Stochastic Symmetry Optimizer) for finding structural isomorphisms.

For details, see [Reasoning Orchestrator and Energy Minimization](#5.2).

---

### CREV Pipeline — Knowledge Extraction
The `CREVPipeline` (Complex Relational Extraction and Validation) is responsible for populating the NSIR graph from external streams. It transforms text or structured data into `RelationalTriplet` objects.

- **Extraction**: Uses `RelationPattern` matching to identify subjects, predicates, and objects.
- **Validation**: Each triplet is assigned an anomaly score and consistency check before being committed to the `KnowledgeGraphIndex`.
- **Quantum Mapping**: Confidence scores from extraction are mapped directly to complex amplitudes in the node's `Qubit` state.

For details, see [CREV Pipeline — Knowledge Extraction and Triplet Management](#5.3).

---

### Quantum Backend Integration
NSIR supports both simulated and hardware-accelerated quantum operations.

- **Hardware**: Integration with **IBM Quantum** via the `ibm_quantum.zig` client, supporting OpenQASM job submission to backends like `ibm_brisbane`.
- **Simulation**: The `RelationalQuantumLogic` engine provides a local simulation of gates (Hadamard, Pauli-X/Y/Z) and measurement-driven state collapse.
- **Adapter**: The `QuantumTaskAdapter` translates graph-based entanglement requests into `QuantumCircuit` batches.

For details, see [Quantum Backend Integration](#5.4).

---

### Code Entity Mapping

This diagram maps high-level NSIR concepts to their specific implementation structs and files within the codebase.

**Logic to Implementation Map**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJOU0lSIExvZ2ljIgogICAgICAgIGRpcmVjdGlvbiBMUgogICAgICAgIEwxWyJHcmFwaCBUb3BvbG9neSJdCiAgICAgICAgTDJbIlF1YW50dW0gTG9naWMiXQogICAgICAgIEwzWyJFeHRyYWN0aW9uIl0KICAgICAgICBMNFsiT3B0aW1pemF0aW9uIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJDb2RlIEVudGl0aWVzIgogICAgICAgIGRpcmVjdGlvbiBMUgogICAgICAgIEUxWyJTZWxmU2ltaWxhclJlbGF0aW9uYWxHcmFwaDxici8+KG5zaXJfY29yZS56aWcpIl0KICAgICAgICBFMlsiUmVsYXRpb25hbFF1YW50dW1Mb2dpYzxici8+KHF1YW50dW1fbG9naWMuemlnKSJdCiAgICAgICAgRTNbIkNSRVZQaXBlbGluZTxici8+KGNyZXZfcGlwZWxpbmUuemlnKSJdCiAgICAgICAgRTRbIkVTU08gT3B0aW1pemVyPGJyLz4oZXNzb19vcHRpbWl6ZXIuemlnKSJdCiAgICBlbmQKCiAgICBMMSAtLT4gRTEKICAgIEwyIC0tPiBFMgogICAgTDMgLS0+IEUzCiAgICBMNCAtLT4gRTQ=)

---

*[Back to Table of Contents](#table-of-contents) | Page 15 of 34 | Next: NSIR Core — Graph Structure and Quantum Operations*

<a id="page-16"></a>

# NSIR Core — Graph Structure and Quantum Operations




The **Non-linear Self-Similar Information Retrieval (NSIR)** system is the relational backbone of JAIDE. At its center is the `SelfSimilarRelationalGraph`, a high-dimensional graph structure where nodes represent information entities and edges represent quantum-correlated relationships. Unlike classical graphs, NSIR utilizes complex-valued amplitudes (Qubits) to represent the state of knowledge, allowing for superposition and entanglement of information.

### 1. Quantum State Representation

NSIR represents node states and relationship strengths using quantum primitives. Every node in the graph contains a `Qubit` representing its activation state, while edges maintain quantum correlations.

#### Qubit and QuantumState
The `Qubit` struct stores two complex amplitudes ($a$ and $b$). The system ensures that the state is normalized such that $|a|^2 + |b|^2 = 1$, where $|a|^2$ represents the probability of the node being in state 0 (inactive/false) and $|b|^2$ represents state 1 (active/true).

The `QuantumState` struct expands this for more complex logic operations, tracking `entanglement_degree` and `phase`.

#### EdgeQuality
Relationships between nodes are categorized by their "coherence" level via the `EdgeQuality` enum:

| Enum Value | Description |
| :--- | :--- |
| `superposition` | Relationship exists in multiple potential states. |
| `entangled` | State of one node is inextricably linked to another. |
| `coherent` | Stable, phase-aligned relationship. |
| `collapsed` | A definite, classical relationship (result of measurement). |
| `fractal` | Self-similar relationship across different scales. |

---

### 2. Graph Lifecycle and Topology

The `SelfSimilarRelationalGraph` manages the lifecycle of knowledge nodes and their interconnects. It supports standard CRUD operations alongside quantum-specific operations like entanglement.

#### Node and Edge Lifecycle
1.  **addNode**: Creates a `Node` with a unique ID, associated data, and an initial `Qubit` state.
2.  **addEdge**: Establishes a connection between a source and target node, assigning a `weight`, `quantum_correlation`, and `fractal_dimension`.
3.  **removeNode**: Deletes a node and all incident edges, ensuring memory is reclaimed via the node's internal allocator.

#### Topology Integrity (SHA-256 Merkle Hash)
To ensure the integrity of the knowledge base, the graph implements a `computeTopologyHash` function. This generates a SHA-256 hash of the entire graph structure by iterating through nodes and edges in a deterministic order, effectively creating a Merkle-style fingerprint of the current state.

**Graph Logic Data Flow:**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJMb2dpYyBTcGFjZSIKICAgICAgICBBWyJMb2dpY0dhdGUgKENOT1QvSEFEQU1BUkQpIl0gLS0+IEJbIlF1YW50dW1TdGF0ZSBFdm9sdXRpb24iXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGUgRW50aXR5IFNwYWNlIgogICAgICAgIEIgLS0+IENbIlNlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoOjplbnRhbmdsZU5vZGVzIl0KICAgICAgICBDIC0tPiBEWyJOb2RlOjpxdWJpdCJdCiAgICAgICAgRCAtLT4gRVsiRWRnZTo6cXVhbnR1bV9jb3JyZWxhdGlvbiJdCiAgICBlbmQKCiAgICBFIC0tPiBGWyJjb21wdXRlVG9wb2xvZ3lIYXNoIChTSEEtMjU2KSJd)
---

### 3. Quantum Operations: Entanglement and Measurement

NSIR facilitates reasoning through quantum gates and state collapse.

*   **entangleNodes**: Links two nodes such that their `Qubit` states become correlated. This is reflected in the `Edge`'s `quantum_correlation` field.
*   **measure**: Collapses a node's `Qubit` from a superposition of states into a classical bit (0 or 1) based on the probability distribution defined by its amplitudes. This operation is irreversible and propagates through the graph, potentially collapsing entangled neighbors.

#### Logic Gates
The system supports a variety of quantum gates via `LogicGate`:
*   **Single-Qubit**: `HADAMARD`, `PAULI_X`, `PHASE`, `FRACTAL_TRANSFORM`.
*   **Multi-Qubit**: `CNOT`, `TOFFOLI`, `RELATIONAL_AND`, `RELATIONAL_XOR`.

---

### 4. Memory Strategy and Performance

The NSIR core is designed for high-throughput relational processing and supports multiple memory allocation strategies via the `core_memory` module.

| Strategy | Implementation | Use Case |
| :--- | :--- | :--- |
| **Arena** | `std.heap.ArenaAllocator` | Short-lived reasoning cycles where the entire graph is discarded. |
| **Pool** | `PoolAllocator` | Fixed-size `Node` and `Edge` allocations to minimize fragmentation. |
| **Buddy** | `BuddyAllocator` | Variable-sized metadata and data buffer allocations. |

#### Data Export
The graph provides utilities to bridge the relational space with the neural (RSF) space:
*   **exportNodeEmbeddings**: Converts node `Qubit` states and metadata into a `Tensor` for neural processing.
*   **exportAdjacencyMatrix**: Generates a weighted adjacency matrix representation of the graph, often used for global topology analysis.

**Relational to Tensor Mapping:**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIC8gUmVsYXRpb25hbCIKICAgICAgICBOb2RlQVsiTm9kZTogJ0NvbmNlcHQgQSciXQogICAgICAgIE5vZGVCWyJOb2RlOiAnQ29uY2VwdCBCJyJdCiAgICAgICAgUmVsWyJFZGdlOiAnRW50YW5nbGVkJyJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdHk6IFNlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoIgogICAgICAgIE5vZGVBIC0tICJFZGdlIHN0cnVjdCIgLS0+IE5vZGVCCiAgICAgICAgTm9kZUEgLS0gIlF1Yml0IiAtLT4gQW1wQVsiQ29tcGxleCBBbXBsaXR1ZGVzIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJOZXVyYWwgU3BhY2UiCiAgICAgICAgQW1wQSAtLSAiZXhwb3J0Tm9kZUVtYmVkZGluZ3MiIC0tPiBUWyJUZW5zb3IgKHNyYy9jb3JlL3RlbnNvci56aWcpIl0KICAgICAgICBSZWwgLS0gImV4cG9ydEFkamFjZW5jeU1hdHJpeCIgLS0+IE1bIkFkamFjZW5jeSBUZW5zb3IiXQogICAgZW5k)
---

### 5. Temporal Dynamics and Signal Propagation

The graph is not static; it supports temporal versioning and active signal propagation.

*   **Temporal Graph**: The `TemporalNode` and `EdgeVersion` structures allow the graph to maintain a history of state changes, indexed by nanosecond timestamps.
*   **Signal Propagation**: The `SignalPropagationEngine` simulates how activations (signals) flow through the graph. Signals have `amplitude`, `phase`, and `frequency`, and their flow is influenced by the `EdgeQuality` and `weight`.


---

*[Back to Table of Contents](#table-of-contents) | Page 16 of 34 | Next: Reasoning Orchestrator and Energy Minimization*

<a id="page-17"></a>

# Reasoning Orchestrator and Energy Minimization




The `ReasoningOrchestrator` is the central coordination engine of the NSIR graph system. It manages the iterative refinement of the `SelfSimilarRelationalGraph` by balancing local node perturbations, global symmetry detection, and meta-level fractal rebalancing. The system operates on the principle of **Energy Minimization**, where the "energy" of the graph represents the contradiction or instability within the relational knowledge base.

## Reasoning Hierarchy: Thought Levels

The orchestrator organizes reasoning into a three-tier hierarchy defined by the `ThoughtLevel` enum. Each level targets different granularities of the graph structure:

| Level | Scope | Primary Operation |
| :--- | :--- | :--- |
| `local` | Individual Nodes/Edges | `perturbLocalNodes`, `updateLocalEdges` |
| `global` | Graph Topology | `esso.optimize`, `transformNodes` |
| `meta` | Structural Integrity | `rebalanceFractalTree`, `chaos_kernel.step` |

### Reasoning Phase Lifecycle
Each reasoning cycle is encapsulated in a `ReasoningPhase`. A phase tracks its own energy delta and determines convergence based on a configurable threshold.

## Energy Minimization and Convergence

The orchestrator aims to reach a "ground state" where the relational graph is internally consistent. This is measured via a graph energy formula that combines quantum state stability and relational quality.

### Graph Energy Calculation
The `calculateGraphEnergy` function aggregates energy from two primary sources:
1.  **Node Potential**: Based on the `Qubit` state of individual nodes.
2.  **Edge Tension**: Derived from `EdgeQuality` and the entanglement between connected nodes.

### Convergence Logic
Convergence is determined in `hasConverged` by calculating the relative change in energy:
$$\Delta E = \frac{|E_{current} - E_{previous}|}{\max(|E_{previous}|, 1.0)}$$
If $\Delta E < convergence\_threshold$, the phase terminates.

## Implementation Flow

The orchestrator integrates the `EntangledStochasticSymmetryOptimizer` (ESSO) and the `ChaosCoreKernel` to drive the graph toward stability.

### System Integration Diagram
This diagram maps the logical reasoning flow to the specific code entities and files responsible for execution.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJSZWFzb25pbmdPcmNoZXN0cmF0b3IgW3NyYy9jb3JlX3JlbGF0aW9uYWwvcmVhc29uaW5nX29yY2hlc3RyYXRvci56aWddIgogICAgICAgIE9SQ0hbIlJlYXNvbmluZ09yY2hlc3RyYXRvci5ydW5SZWFzb25pbmdDeWNsZSgpIl0KICAgICAgICBTVEVQWyJSZWFzb25pbmdPcmNoZXN0cmF0b3Iuc3RlcCgpIl0KICAgICAgICBMT0NBTFsicGVydHVyYkxvY2FsTm9kZXMoKSJdCiAgICAgICAgRURHRVsidXBkYXRlTG9jYWxFZGdlcygpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJPcHRpbWl6YXRpb24gRW5naW5lcyIKICAgICAgICBFU1NPWyJFU1NPOiBFbnRhbmdsZWRTdG9jaGFzdGljU3ltbWV0cnlPcHRpbWl6ZXIgW3NyYy9jb3JlX3JlbGF0aW9uYWwvZXNzb19vcHRpbWl6ZXIuemlnXSJdCiAgICAgICAgQ0hBT1NbIkNoYW9zQ29yZUtlcm5lbCBbc3JjL2NvcmVfcmVsYXRpb25hbC9jaGFvc19jb3JlLnppZ10iXQogICAgICAgIEZSQUNUQUxbIkZyYWN0YWxUcmVlIFtzcmMvY29yZV9yZWxhdGlvbmFsL2ZuZHMuemlnXSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiRGF0YSBFbnRpdGllcyBbc3JjL2NvcmVfcmVsYXRpb25hbC9uc2lyX2NvcmUuemlnXSIKICAgICAgICBHUkFQSFsiU2VsZlNpbWlsYXJSZWxhdGlvbmFsR3JhcGgiXQogICAgICAgIE5PREVbIk5vZGUgKFF1Yml0IFN0YXRlKSJdCiAgICAgICAgUkVMWyJFZGdlIChFZGdlUXVhbGl0eSkiXQogICAgZW5kCgogICAgT1JDSCAtLT4gU1RFUAogICAgU1RFUCAtLT4gTE9DQUwKICAgIFNURVAgLS0+IEVER0UKICAgIFNURVAgLS0gIkdsb2JhbCBPcHRpbWl6YXRpb24iIC0tPiBFU1NPCiAgICBTVEVQIC0tICJFbnRyb3B5IE1hbmFnZW1lbnQiIC0tPiBDSEFPUwogICAgU1RFUCAtLSAiU3RydWN0dXJhbCBCYWxhbmNpbmciIC0tPiBGUkFDVEFMCiAgICAKICAgIExPQ0FMIC0tPiBOT0RFCiAgICBFREdFIC0tPiBSRUwKICAgIEVTU08gLS0+IEdSQVBI)
## Key Orchestration Functions

### 1. Symmetry Detection via ESSO
The orchestrator calls the `EntangledStochasticSymmetryOptimizer` to find repeating patterns or structural symmetries in the graph. ESSO uses `SymmetryGroup` types (reflection, rotation, translation) to identify redundant information or "canonical" relational structures.

### 2. Local Perturbations
*   **`perturbLocalNodes`**: Randomly adjusts the quantum amplitudes of a subset of nodes to escape local energy minima.
*   **`updateLocalEdges`**: Refines `EdgeQuality` based on the current state of source and target nodes.

### 3. ChaosCoreKernel Cycles
The `ChaosCoreKernel` manages the underlying memory and task load. During a reasoning cycle, the orchestrator triggers `chaos_kernel.step()` to perform:
*   **Memory Balancing**: Reallocating `MemoryBlock` entities across cores.
*   **Entanglement Cleanup**: Removing stale or low-priority entanglements between memory blocks.

## Data Flow: Reasoning to Graph State

The following diagram illustrates how high-level reasoning instructions translate into modifications of the NSIR graph primitives.

![Diagram](https://mermaid.ink/img/c2VxdWVuY2VEaWFncmFtCiAgICBwYXJ0aWNpcGFudCBSTyBhcyBSZWFzb25pbmdPcmNoZXN0cmF0b3IKICAgIHBhcnRpY2lwYW50IEVTU08gYXMgRVNTT19PcHRpbWl6ZXIKICAgIHBhcnRpY2lwYW50IEdSQVBIIGFzIFNlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoCiAgICBwYXJ0aWNpcGFudCBOT0RFIGFzIE5vZGUvUXViaXQKCiAgICBSTy0+PkVTU086IG9wdGltaXplKGdyYXBoLCBpdGVyYXRpb25zKQogICAgTm90ZSBvdmVyIEVTU086IERldGVjdCBTeW1tZXRyeVBhdHRlcm5zIFtzcmMvY29yZV9yZWxhdGlvbmFsL2Vzc29fb3B0aW1pemVyLnppZ10KICAgIEVTU08tPj5HUkFQSDogYXBwbHlUcmFuc2Zvcm0oU3ltbWV0cnlUcmFuc2Zvcm0pCiAgICBHUkFQSC0+Pk5PREU6IHVwZGF0ZUNvbXBsZXhBbXBsaXR1ZGVzKCkKICAgIFJPLT4+Uk86IGNhbGN1bGF0ZUdyYXBoRW5lcmd5KCkKICAgIE5vdGUgcmlnaHQgb2YgUk86IENoZWNrIGNvbnZlcmdlbmNlX3RocmVzaG9sZCBbc3JjL2NvcmVfcmVsYXRpb25hbC9yZWFzb25pbmdfb3JjaGVzdHJhdG9yLnppZzo0N10KICAgIFJPLT4+Uk86IHJlY29yZFBoYXNlKE9yY2hlc3RyYXRvclN0YXRpc3RpY3Mp)
## Orchestrator Statistics

The `OrchestratorStatistics` struct provides telemetry for the reasoning process:

*   **`best_energy_achieved`**: The lowest energy state found across all phases.
*   **`average_convergence_time`**: Moving average of nanoseconds required to reach the convergence threshold.
*   **`patterns_discovered`**: Total number of `SymmetryPattern` objects identified by ESSO and recorded in the `ReasoningPhase`.

The `recordPhase` function updates these metrics at the end of every `ThoughtLevel` execution.


---

*[Back to Table of Contents](#table-of-contents) | Page 17 of 34 | Next: CREV Pipeline — Knowledge Extraction and Triplet Management*

<a id="page-18"></a>

# CREV Pipeline — Knowledge Extraction and Triplet Management




The **CREV (Categorical Relational Extraction and Validation) Pipeline** is a five-stage ingestion and refinement engine designed to transform unstructured text, structured data streams, and image metadata into a high-fidelity Knowledge Graph. It serves as the primary bridge between raw input and the **NSIR (Non-linear Self-Similar Information Retrieval)** graph system, ensuring that all extracted information is validated for consistency and mapped into a quantum-relational state.

### Pipeline Architecture and Data Flow

The `CREVPipeline` orchestrates the lifecycle of information from initial ingestion to its final representation in the `KnowledgeGraphIndex`. It utilizes a staged approach to manage complexity and ensure data integrity.

#### The Five Stages of Extraction
The pipeline follows a strict linear progression defined by the `ExtractionStage` enum:
1.  **Tokenization**: Initial stream processing and morpheme-guided segmentation.
2.  **Triplet Extraction**: Identification of Subject-Relation-Object patterns.
3.  **Validation**: Anomaly scoring and consistency checking against existing knowledge.
4.  **Integration**: Conflict resolution and merging with the `SelfSimilarRelationalGraph`.
5.  **Indexing**: Optimization for retrieval via the `KnowledgeGraphIndex`.

**Diagram: CREV Pipeline Processing Flow**
![CREV Pipeline Processing Flow](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJJbmdlc3Rpb24gTGF5ZXIiCiAgICAgICAgIlN0cmVhbVtUZXh0L1N0cnVjdHVyZWQvSW1hZ2VdIiAtLT4gIkNSRVZQaXBlbGluZS5wcm9jZXNzU3RyZWFtKCkiCiAgICBlbmQKCiAgICBzdWJncmFwaCAiRXh0cmFjdGlvbiBTdGFnZXMgW0V4dHJhY3Rpb25TdGFnZV0iCiAgICAgICAgIkNSRVZQaXBlbGluZS5wcm9jZXNzU3RyZWFtKCkiIC0tPiAiU3RhZ2U6OnRva2VuaXphdGlvbiIKICAgICAgICAiU3RhZ2U6OnRva2VuaXphdGlvbiIgLS0+ICJTdGFnZTo6dHJpcGxldF9leHRyYWN0aW9uIgogICAgICAgICJTdGFnZTo6dHJpcGxldF9leHRyYWN0aW9uIiAtLT4gIlN0YWdlOjp2YWxpZGF0aW9uIgogICAgICAgICJTdGFnZTo6dmFsaWRhdGlvbiIgLS0+ICJTdGFnZTo6aW50ZWdyYXRpb24iCiAgICAgICAgIlN0YWdlOjppbnRlZ3JhdGlvbiIgLS0+ICJTdGFnZTo6aW5kZXhpbmciCiAgICBlbmQKCiAgICBzdWJncmFwaCAiU3RvcmFnZSAmIEZlZWRiYWNrIgogICAgICAgICJTdGFnZTo6aW5kZXhpbmciIC0tPiAiS25vd2xlZGdlR3JhcGhJbmRleCIKICAgICAgICAiU3RhZ2U6OmludGVncmF0aW9uIiAtLT4gIlNlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoIgogICAgICAgICJDaGFvc0NvcmVLZXJuZWwiIC0uLT4gfCJQZXJ0dXJiYXRpb24ifCAiU3RhZ2U6OnZhbGlkYXRpb24iCiAgICBlbmQKCiAgICAiUmVsYXRpb25hbFRyaXBsZXQiIC0tICJVc2VkIGJ5IiAtLT4gIkV4dHJhY3Rpb24gU3RhZ2VzIFtFeHRyYWN0aW9uU3RhZ2VdIg==)
---

### Relational Triplet Management

The fundamental unit of knowledge in the CREV pipeline is the `RelationalTriplet`. Unlike standard RDF triplets, these structures carry high-dimensional metadata, including confidence scores and temporal anchors.

#### Data Structure: `RelationalTriplet`
A `RelationalTriplet` consists of:
*   **Core Entities**: `subject`, `relation`, and `object` (stored as `[]u8`).
*   **Confidence**: A `f64` value clamped between `0.0` and `1.0` ``.
*   **Identity Hash**: A SHA-256 hash generated via `hashTripletIdentity` to prevent duplicate ingestion of identical semantic relations ``.
*   **Metadata**: A `StringHashMap(`.

**Code Entity Mapping: Triplet Identity**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgIFNbIidBcHBsZSciXQogICAgICAgIFJbIidpc19hJyJdCiAgICAgICAgT1siJ0ZydWl0JyJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdHkgU3BhY2UiCiAgICAgICAgUlRbIlJlbGF0aW9uYWxUcmlwbGV0IChzdHJ1Y3QpIl0KICAgICAgICBIVElbImhhc2hUcmlwbGV0SWRlbnRpdHkgKGZuKSJdCiAgICAgICAgU0hbInNvdXJjZV9oYXNoIChbMzJddTgpIl0KICAgIGVuZAoKICAgIFMgLS0+IFJUCiAgICBSIC0tPiBSVAogICAgTyAtLT4gUlQKICAgIFJUIC0tPiBIVEkKICAgIEhUSSAtLT4gU0g=)
---

### Validation and Conflict Resolution

Before a triplet is integrated into the NSIR graph, it must pass the `validation` stage. This involves two primary checks:
1.  **Anomaly Scoring**: Comparing the new triplet against the `ChaosCoreKernel` to determine if the relation represents a statistically improbable edge.
2.  **Consistency Check**: Verifying if the new information contradicts high-confidence triplets already in the `KnowledgeGraphIndex`.

#### Quantum-Relational Mapping
During the `integration` stage, the `confidence` of a `RelationalTriplet` is mapped to a complex amplitude within the `SelfSimilarRelationalGraph`. This allows the system to treat uncertainty as a quantum superposition of states.
*   **High Confidence**: Results in a collapsed state with high `EdgeQuality`.
*   **Low Confidence**: Maintains a high "energy" state, subject to frequent perturbation by the `ChaosCoreKernel`.

---

### KnowledgeGraphIndex (Three-Axis Indexing)

The `KnowledgeGraphIndex` provides O(1) or O(log N) lookup for triplets based on any of the three axes (Subject, Relation, or Object). It is implemented using a nested `StringHashMap` structure.

| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Primary Index** | `StringHashMap(ArrayList(*RelationalTriplet))` | Maps a subject to all its associated relations. |
| **Relation Index** | `StringHashMap(ArrayList(*RelationalTriplet))` | Groups all triplets sharing a specific relation type (e.g., "is_part_of"). |
| **Object Index** | `StringHashMap(ArrayList(*RelationalTriplet))` | Reverse lookup from object to subject. |

#### ChaosCoreKernel Integration
The `ChaosCoreKernel` interacts with the pipeline by triggering "re-validation" cycles. If the global graph energy exceeds a certain threshold, the kernel forces the `CREVPipeline` to re-evaluate triplets with low confidence scores, potentially leading to the pruning of "noisy" knowledge.

---

### Technical Implementation Details

#### Triplet Hashing
The system distinguishes between **Identity Hashes** (based on S-R-O strings) and **Field Hashes** (which include confidence and extraction time).

*   **`hashTripletIdentity`**: Used for deduplication. ``
*   **`hashTripletFields`**: Used for auditing and versioning of specific extraction events. ``

#### Memory Management
The `CREVPipeline` utilizes a per-request `ArenaAllocator` for the extraction stages, but promotes `RelationalTriplet` data to a long-lived `SlabAllocator` or `Pool` when integrated into the `KnowledgeGraphIndex`.


---

*[Back to Table of Contents](#table-of-contents) | Page 18 of 34 | Next: Quantum Backend Integration*

<a id="page-19"></a>

# Quantum Backend Integration




The Quantum Backend Integration layer provides the interface between the JAIDE NSIR graph system and quantum computing resources. It supports both high-fidelity simulation via the `RelationalQuantumLogic` engine and physical hardware execution through the IBM Quantum cloud service. This subsystem is responsible for identifying highly entangled subgraphs, translating relational logic into OpenQASM, and managing the lifecycle of quantum jobs.

## System Architecture

The integration is structured as a multi-tier abstraction starting from low-level gate logic up to high-level graph task adaptation.

### Quantum Integration Flow
This diagram illustrates how the `QuantumTaskAdapter` bridges the "Natural Language Space" (represented by relational triplets in the NSIR graph) to the "Code Entity Space" of quantum hardware and simulators.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOU0lSIEdyYXBoIFNwYWNlIgogICAgICAgIEFbIlNlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoIl0gLS0gImNvbnRhaW5zIiAtLT4gQlsiTm9kZSJdCiAgICAgICAgQSAtLSAiY29udGFpbnMiIC0tPiBDWyJFZGdlIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJRdWFudHVtIFRhc2sgQWRhcHRlciBbc3JjL2NvcmVfcmVsYXRpb25hbC9xdWFudHVtX3Rhc2tfYWRhcHRlci56aWddIgogICAgICAgIERbIlF1YW50dW1UYXNrQWRhcHRlci5pZGVudGlmeVF1YW50dW1TdWJncmFwaHMoKSJdCiAgICAgICAgRVsiUXVhbnR1bVN1YmdyYXBoLmlzUXVhbnR1bVN1aXRhYmxlKCkiXQogICAgICAgIEZbIlF1YW50dW1UYXNrQWRhcHRlci5leGVjdXRlVGFzaygpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJCYWNrZW5kIEV4ZWN1dGlvbiIKICAgICAgICBHWyJSZWxhdGlvbmFsUXVhbnR1bUxvZ2ljIChTaW11bGF0b3IpIl0KICAgICAgICBIWyJJQk1RdWFudHVtQ2xpZW50IChIYXJkd2FyZSkiXQogICAgZW5kCgogICAgQiAmIEMgLS0+IEQKICAgIEQgLS0+IEUKICAgIEUgLS0gIlRydWUiIC0tPiBGCiAgICBGIC0tICJ1c2VfcmVhbF9iYWNrZW5kPWZhbHNlIiAtLT4gRwogICAgRiAtLSAidXNlX3JlYWxfYmFja2VuZD10cnVlIiAtLT4gSAogICAgRyAmIEggLS0gIlF1YW50dW1UYXNrUmVzdWx0IiAtLT4gQQ==)
## IBM Quantum Client

The `IBMQuantumClient` manages authenticated communication with IBM Quantum services using Cloud Resource Names (CRN) and API tokens. It targets the `ibm_brisbane` backend by default, submitting jobs formatted in OpenQASM.

### Key Components
*   **Authentication**: Uses a Bearer token and CRN (Cloud Resource Name) retrieved from the `IBM_QUANTUM_CRN` environment variable or manual override.
*   **Job Submission**: The `submitJob` function serializes OpenQASM strings into a JSON payload with a default configuration of 1024 shots.
*   **Result Retrieval**: Polls the IBM Cloud API for job status and retrieves the final measurement bitstrings.

### Backend Hardware Specs
The system maintains a registry of hardware specifications for various IBM architectures (Heron, Eagle, Falcon, Osprey, Condor) to assist in error modeling and qubit allocation.

| Backend Type | Qubit Count | T1 Mean (ns) | Readout Error Mean |
| :--- | :--- | :--- | :--- |
| **Heron** | 133 | 350,000.0 | 0.008 |
| **Eagle** | 127 | 200,000.0 | 0.015 |
| **Falcon** | 27 | 100,000.0 | 0.020 |
| **Simulator**| 32 | N/A | 0.001 |

## Relational Quantum Logic (Simulation Engine)

The `RelationalQuantumLogic` engine provides a local simulation of quantum circuits, specifically optimized for relational operations. It manages `QuantumState` objects which track complex amplitudes, phases, and entanglement degrees.

### Logic Gates
The engine supports standard quantum gates and specialized relational gates used for graph reasoning:
*   **Standard**: `HADAMARD`, `PAULI_X/Y/Z`, `CNOT`, `TOFFOLI`.
*   **Relational**: `RELATIONAL_AND`, `RELATIONAL_OR`, `RELATIONAL_NOT`, `RELATIONAL_XOR`.
*   **Fractal**: `FRACTAL_TRANSFORM` used for scaling self-similar information.

### Quantum State Implementation
A `QuantumState` is represented by two complex amplitudes (for the $|0\rangle$ and $|1\rangle$ basis states).
*   **Normalization**: Ensures the total probability $| \alpha |^2 + | \beta |^2 = 1$.
*   **Measurement**: Probabilistic collapse of the state based on `prob0()` and `prob1()` calculations.

## Quantum Task Adapter

The `QuantumTaskAdapter` acts as the orchestrator for the NSIR graph. It identifies subgraphs that would benefit from quantum processing based on their entanglement and fractal dimensionality.

### Subgraph Identification Logic
The adapter iterates through graph edges and groups nodes into a `QuantumSubgraph` if they exceed defined thresholds:
1.  **Entanglement Threshold**: Default 0.5.
2.  **Fractal Threshold**: Default 1.5.

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJUYXNrIFNlbGVjdGlvbiBbc3JjL2NvcmVfcmVsYXRpb25hbC9xdWFudHVtX3Rhc2tfYWRhcHRlci56aWddIgogICAgICAgIGRpcmVjdGlvbiBUQgogICAgICAgIEVudHJ5WyJpZGVudGlmeVF1YW50dW1TdWJncmFwaHMoKSJdIC0tPiBJdGVyWyJJdGVyYXRlIGdyYXBoLmVkZ2VzIl0KICAgICAgICBJdGVyIC0tPiBDaGVja3siY29ycmVsYXRpb24gPiB0aHJlc2hvbGRcbkFORFxuZGltID4gZnJhY3RhbF90aHJlc2hvbGQifQogICAgICAgIENoZWNrIC0tICJZZXMiIC0tPiBDbHVzdGVyWyJBZGQgdG8gUXVhbnR1bVN1YmdyYXBoIl0KICAgICAgICBDaGVjayAtLSAiTm8iIC0tPiBTa2lwWyJJZ25vcmUgRWRnZSJdCiAgICAgICAgQ2x1c3RlciAtLT4gTWV0cmljc1siY29tcHV0ZU1ldHJpY3MoKSJdCiAgICBlbmQ=)

### Data Flow: Execution to Result
When a task is executed via `executeTask`, the adapter:
1.  Translates the `QuantumSubgraph` into a series of `LogicGate` operations.
2.  Submits to either the `local_simulator` or the `quantum_client`.
3.  Wraps the output in a `QuantumTaskResult` containing complex amplitudes and execution statistics.

## Configuration and Constants

The `QuantumConfig` struct defines the operational limits for both simulation and hardware backends.

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `MAX_QUBITS_SIMULATION` | 20 | Limits memory usage for local state vectors |
| `HARDWARE_MAX_SHOTS` | 100,000 | Maximum sampling rate for IBM hardware |
| `SIMULATOR_QUBITS` | 32 | Maximum addressable qubits in simulator |
| `DEFAULT_SHOTS` | 4,000 | Default sampling for statistical convergence |
| `POLL_INTERVAL_MS` | 100 | Wait time between job status checks |


---

*[Back to Table of Contents](#table-of-contents) | Page 19 of 34 | Next: Optimization and Training*

<a id="page-20"></a>

# Optimization and Training




This section provides an overview of the JAIDE v40 training stack, which integrates a high-performance second-order optimizer, a distributed training harness for multi-GPU scaling, and cloud-native orchestration via Modal. The system is designed to leverage **B200 GPU** architectures and **NCCL** for high-throughput model convergence.

## Training Stack Overview

The JAIDE training pipeline is built on three main pillars:
1.  **SFD Optimizer**: A sophisticated second-order optimization engine implementing Stochastic Fisher Diagonal updates and SophiaSOAP preconditioning.
2.  **Distributed Harness**: A weight-delta averaging system that coordinates multiple GPU workers using NCCL collective operations.
3.  **Cloud Orchestration**: Python-based scripts for Modal that automate environment provisioning, dataset ingestion, and multi-node execution.

### System Architecture Diagram

The following diagram illustrates the relationship between the training components and the underlying hardware abstraction.

**Training System Interaction**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJDbG91ZCAvIE1vZGFsIFNwYWNlIgogICAgICAgIE1bIm1vZGFsX3RyYWluLnB5Il0gLS0gIm9yY2hlc3RyYXRlcyIgLS0+IERUWyJEaXN0cmlidXRlZFRyYWluZXJGdXRoYXJrIl0KICAgICAgICBNIC0tICJwcm92aXNpb25zIiAtLT4gQjIwMFsiOHggQjIwMCBHUFUgQ2x1c3RlciJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiRGlzdHJpYnV0ZWQgVHJhaW5pbmcgSGFybmVzcyIKICAgICAgICBEVCAtLSAibWFuYWdlcyIgLS0+IEdDWyJHUFVDb29yZGluYXRvciJdCiAgICAgICAgR0MgLS0gImNhbGxzIiAtLT4gTkNDTFsiTkNDTCAoYWxsUmVkdWNlL2Jyb2FkY2FzdCkiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIk9wdGltaXphdGlvbiBDb3JlIgogICAgICAgIERUIC0tICJzdGVwcyIgLS0+IFNGRFsiU0ZEIE9wdGltaXplciJdCiAgICAgICAgU0ZEIC0tICJ1c2VzIiAtLT4gTVBUWyJNaXhlZFByZWNpc2lvblRyYWluZXIiXQogICAgICAgIE1QVCAtLSAia2VybmVscyIgLS0+IEZLWyJGdXRoYXJrIEtlcm5lbHMiXQogICAgZW5kCgogICAgRksgLS0gImV4ZWN1dGVzIG9uIiAtLT4gQjIwMA==)
## Component Breakdown

### SFD Optimizer — Second-Order Training
The **Stochastic Fisher Diagonal (SFD)** optimizer is the primary engine for model convergence. Unlike standard first-order methods (SGD/Adam), SFD utilizes second-order information to navigate the loss landscape more efficiently.

*   **Key Features**:
    *   **SophiaSOAP**: Implements KFAC preconditioning and Hutchinson Hessian estimation for second-order curvature correction.
    *   **Mixed Precision**: A `MixedPrecisionTrainer` supports quantization levels from **FP32** down to **FP4** for memory-efficient training on large models.
    *   **Memory Management**: Specialized `B200MemoryManager` (TMEM) support for high-bandwidth memory utilization.

For details, see [SFD Optimizer — Second-Order Training](#6.1).

### Distributed Training
The distributed harness enables JAIDE to scale across multiple GPUs and nodes. It follows a weight-delta averaging pattern to maintain consistency across the cluster.

*   **Key Components**:
    *   **DistributedTrainerFuthark**: The main entry point for GPU-accelerated training, handling dataset partitioning and local batch processing.
    *   **GPUCoordinator**: Manages the **NCCL** lifecycle, providing primitives like `allReduce`, `broadcast`, and `barrier` to synchronize model states.
    *   **Dataset Partitioning**: Automatically handles JSONL stream splitting to ensure each worker processes unique samples.

For details, see [Distributed Training](#6.2).

### Cloud Training with Modal
To simplify the deployment of 8×B200 clusters, JAIDE includes a comprehensive Modal-based cloud stack.

*   **Workflow**:
    *   **Image Build**: A custom Ubuntu-based image containing **Zig 0.13.0**, **Futhark**, and **CUDA 12.4**.
    *   **Dataset Ingestion**: Automatically downloads and converts the `finephrase` dataset into a training-ready JSONL format.
    *   **Runtime Compilation**: If pre-built binaries are missing, the script triggers a fallback build of Futhark kernels and the Zig executable on the remote worker.

For details, see [Cloud Training with Modal](#6.3).

## Training Logic Flow

The following diagram maps the logical flow from the Python orchestration layer down to the Zig-based training loop.

**Code Entity Mapping: Orchestration to Execution**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJQeXRob24gKG1vZGFsX3RyYWluLnB5KSIKICAgICAgICBTVEFSVFsiQGFwcC5mdW5jdGlvbiJdIC0tPiBSVU5bInRyYWluKCkiXQogICAgICAgIFJVTiAtLT4gU1VCWyJzdWJwcm9jZXNzLnJ1bihqYWlkZS1kaXN0cmlidXRlZCkiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlppZyAoZGlzdHJpYnV0ZWRfdHJhaW5lcl9mdXRoYXJrLnppZykiCiAgICAgICAgU1VCIC0tPiBJTklUWyJEaXN0cmlidXRlZFRyYWluZXJGdXRoYXJrLmluaXQiXQogICAgICAgIElOSVQgLS0+IExPT1BbInRyYWluT25lRXBvY2goKSJdCiAgICAgICAgTE9PUCAtLT4gU1RFUFsic3RlcCgpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJPcHRpbWl6YXRpb24gKHNmZC56aWcpIgogICAgICAgIFNURVAgLS0+IE9QVFsiU0ZELnVwZGF0ZSgpIl0KICAgICAgICBPUFQgLS0+IFBSRUNbIk1peGVkUHJlY2lzaW9uVHJhaW5lci5xdWFudGl6ZSJdCiAgICBlbmQ=)
## Hyperparameter Configuration

Training is governed by the `TrainerConfig` and `TrainingParameters` structures, which define the learning rate, momentum, and architectural constraints.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | `f32` | `0.001` | Base step size for updates |
| `momentum` | `f32` | `0.0` | Momentum factor for SFD |
| `max_line_size` | `usize` | `10MB` | Max buffer for dataset JSONL lines |
| `checkpoint_version`| `u32` | `4` | Versioning for RSF model persistence |


---

*[Back to Table of Contents](#table-of-contents) | Page 20 of 34 | Next: SFD Optimizer — Second-Order Training*

<a id="page-21"></a>

# SFD Optimizer — Second-Order Training




The **Stochastic Fisher Diagonal (SFD)** optimizer is JAIDE's primary second-order training engine. It combines elements of natural gradient descent via Fisher information preconditioning with advanced variance reduction and mixed-precision techniques to enable efficient training of high-dimensional RSF architectures.

## Core Implementation and Data Flow

The SFD optimizer manages the update lifecycle of model parameters by tracking first-order momentum (velocity) and second-order curvature estimates (Fisher diagonal). It utilizes a preconditioned gradient approach, inspired by Sophia and SOAP, to adapt learning rates based on the local geometry of the loss surface.

### SFD State Management
The `SFD` struct maintains the optimizer's state, including hyperparameters and buffers for momentum and curvature.

| Component | Code Entity | Description |
| :--- | :--- | :--- |
| **Momentum** | `velocity` | Tracks the exponentially weighted moving average of gradients. |
| **Fisher Diagonal** | `fisher_diag` | Tracks the moving average of squared gradients (or Hessian diagonals). |
| **Preconditioning** | `SophiaSOAP` | Implements KFAC-style preconditioning and Hutchinson Hessian estimation. |
| **Variance Reduction** | `MARS` | Reduces gradient noise in stochastic settings. |

### System Data Flow
The following diagram illustrates the flow of gradients through the SFD optimization pipeline, from raw backpropagation outputs to quantized parameter updates.

**Optimizer Data Flow: Gradient to Parameter Update**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJHcmFkaWVudCBQcm9jZXNzaW5nIgogICAgICAgIEdbIlJhdyBHcmFkaWVudHMgKGYzMikiXSAtLT4gTUFSU1siTUFSUyBWYXJpYW5jZSBSZWR1Y2VyIl0KICAgICAgICBNQVJTIC0tPiBWWyJWZWxvY2l0eSBVcGRhdGUgKE1vbWVudHVtKSJdCiAgICAgICAgRyAtLT4gRkRbIkZpc2hlciBEaWFnb25hbCBVcGRhdGUiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlByZWNvbmRpdGlvbmluZyIKICAgICAgICBGRCAtLT4gS0ZBQ1siS0ZBQyBQcmVjb25kaXRpb25lciJdCiAgICAgICAgViAtLT4gU09BUFsiU29waGlhU09BUCBQcmVjb25kaXRpb25pbmciXQogICAgICAgIEtGQUMgLS0+IFNPQVAKICAgIGVuZAoKICAgIHN1YmdyYXBoICJVcGRhdGUgRXhlY3V0aW9uIgogICAgICAgIFNPQVAgLS0+IFN0ZXBbIk9wdGltaXplciBTdGVwIENhbGN1bGF0aW9uIl0KICAgICAgICBTdGVwIC0tPiBMUlNbIkxSU2NoZWR1bGVyIChEZWNheS9XYXJtdXApIl0KICAgICAgICBMUlMgLS0+IFF1YW50WyJNaXhlZFByZWNpc2lvblRyYWluZXIgKFF1YW50aXphdGlvbikiXQogICAgICAgIFF1YW50IC0tPiBQWyJVcGRhdGVkIFBhcmFtZXRlcnMgKGYzMi9mcDE2L2ZwOCkiXQogICAgZW5kCgogICAgc3R5bGUgRyBzdHJva2UtZGFzaGFycmF5OiA1IDUKICAgIHN0eWxlIFAgc3Ryb2tlLXdpZHRoOiA0cHg=)
---

## Key Components

### 1. Stochastic Fisher Diagonal (SFD)
The `SFD` class is the central coordinator for the training loop. It implements the update rule:
$$\theta_{t+1} = \theta_t - \eta \cdot \text{Preconditioner}(m_t, \hat{F}_t)$$
where $m_t$ is the momentum and $\hat{F}_t$ is the diagonal Fisher estimate.

*   **Initialization**: `init(allocator, params, options)` sets up velocity and Fisher buffers.
*   **Update Step**: `step()` applies the calculated gradients to the parameters.

### 2. SophiaSOAP and KFAC Preconditioning
The optimizer implements a hybrid of **Sophia** (Second-order Stochastic Optimization) and **SOAP** (Shampoo with Optimal Any-order Preconditioning). It uses **Hutchinson's method** to estimate the Hessian diagonal without explicit matrix computation.

*   **Hutchinson Estimation**: Uses Rademacher noise vectors to approximate $diag(H)$.
*   **KFAC Preconditioning**: Approximates the Fisher Information Matrix as a block-diagonal matrix for efficient inversion.

### 3. MixedPrecisionTrainer
To support high-performance hardware like the NVIDIA B200, SFD supports multiple precision formats. The `quantizeValue` function handles the mapping of `f32` gradients to lower-precision representations.

| Precision | Implementation Details | Range |
| :--- | :--- | :--- |
| **FP4** | 3-bit mantissa, 1-bit sign, custom levels | [-6.0, 6.0] |
| **FP8** | E4M3/E5M2 style quantization | [-448.0, 448.0] |
| **FP16** | Standard half-precision approximation | [-65504, 65504] |

---

## Bayesian Optimization and LR Scheduling

The training process is governed by a `BayesianOptimizer` that tunes hyperparameters (like learning rate and weight decay) using a `GaussianProcess` surrogate model.

### Bayesian Tuning Pipeline
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJCYXllc2lhbiBMb29wIgogICAgICAgIEdQWyJHYXVzc2lhblByb2Nlc3MgU3Vycm9nYXRlIl0gLS0+IEFDUVsiQWNxdWlzaXRpb24gRnVuY3Rpb24gKEVJL1VDQikiXQogICAgICAgIEFDUSAtLT4gRXZhbFsiRXZhbHVhdGUgSHlwZXJwYXJhbXMiXQogICAgICAgIEV2YWwgLS0+IFJlc3VsdFsiVHJhaW5pbmcgTG9zcy9NZXRyaWMiXQogICAgICAgIFJlc3VsdCAtLT4gR1AKICAgIGVuZAoKICAgIHN1YmdyYXBoICJFeGVjdXRpb24iCiAgICAgICAgRXZhbCAtLT4gTFJTWyJMUlNjaGVkdWxlciJdCiAgICAgICAgTFJTIC0tPiBTRkRbIlNGRC5zdGVwKCkiXQogICAgZW5k)
### LRScheduler
The `LRScheduler` supports multiple regimes:
*   **Linear Warmup**: Gradually increases LR from zero to the target.
*   **Cosine Decay**: Reduces LR following a cosine curve to ensure convergence.

---

## Hardware Integration: B200 and Kernel Fusion

The SFD optimizer is optimized for modern GPU architectures through the `B200MemoryManager` and kernel fusion strategies.

### B200MemoryManager (TMEM)
This component manages the **Tensor Memory (TMEM)** available on Blackwell-class GPUs. It ensures that velocity and Fisher buffers are localized to fast on-chip memory to minimize HBM bandwidth bottlenecks.
*   **Allocation**: `allocateTMEM(size)` reserves blocks in the hardware-accelerated tensor memory pool.

### Kernel Fusion
SFD performs "kernel fusion" by combining the momentum update, Fisher diagonal update, and parameter application into a single GPU kernel pass. This reduces the number of memory round-trips (loads/stores) per optimizer step.

**Code Entity Mapping**
| System Concept | Zig Class/Struct | File Reference |
| :--- | :--- | :--- |
| **Optimizer Core** | `SFD` | |
| **Tensor Data** | `Tensor` | |
| **Memory Manager** | `B200MemoryManager` | |
| **Variance Reducer** | `MARS` | |
| **Preconditioner** | `SophiaSOAP` | |


---

*[Back to Table of Contents](#table-of-contents) | Page 21 of 34 | Next: Distributed Training*

<a id="page-22"></a>

# Distributed Training




The distributed training subsystem provides the infrastructure for multi-GPU and hybrid quantum-classical model optimization. It leverages **NCCL** (NVIDIA Collective Communications Library) for high-performance GPU-to-GPU communication and **Futhark** for accelerated kernel execution. The system supports weight-delta averaging across ranks, dataset partitioning, and synchronous barrier primitives.

## Architecture and Coordination

The `GPUCoordinator` serves as the primary interface for managing distributed state and collective operations. It initializes the NCCL communicator, manages CUDA streams, and provides abstraction for standard collective operations used during the training loop.

### GPUCoordinator Initialization
The initialization process involves several steps to ensure all ranks are synchronized and mapped to the correct hardware:
1.  **Device Selection**: Ranks are mapped to local GPUs using `rank % local_device_count`.
2.  **NCCL Setup**: A `ncclUniqueId` is shared among all ranks (typically via a shared file system) and used to initialize the `ncclComm`.
3.  **Stream Creation**: A dedicated `cudaStream_t` is created for overlapping communication with computation.
4.  **Barrier Allocation**: A small GPU buffer is allocated to facilitate the `barrier()` implementation.

### Collective Operations
The coordinator wraps NCCL primitives to handle data movement across the `world_size`:
*   **`allReduce`**: Aggregates tensors across all GPUs (e.g., for gradient averaging).
*   **`broadcast`**: Synchronizes weights from the root rank (Rank 0) to all other ranks.
*   **`allGather`**: Collects partial results from all ranks into a single large buffer.
*   **`reduceScatter`**: Reduces data and scatters the result across ranks.

### Barrier Implementation
The `barrier()` function ensures all ranks reach a specific execution point before proceeding. It uses an `allReduce` operation on a dummy `barrier_buffer` to force synchronization across the NCCL communicator.

### Data Flow: Distributed Coordination
The following diagram illustrates the relationship between the high-level `DistributedTrainer` and the underlying NCCL/CUDA primitives.

Title: Distributed Coordination Data Flow
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJIb3N0IFNwYWNlIChaaWcpIgogICAgICAgIEFbIm1haW5fZGlzdHJpYnV0ZWQuemlnIl0gLS0gImluaXQoKSIgLS0+IEJbIkdQVUNvb3JkaW5hdG9yIl0KICAgICAgICBCIC0tICJuY2NsR2V0VW5pcXVlSWQiIC0tPiBDWyJuY2NsX2JpbmRpbmdzLnppZyJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiR1BVIFNwYWNlIChDVURBL05DQ0wpIgogICAgICAgIEIgLS0gImN1ZGFTdHJlYW1DcmVhdGUiIC0tPiBEWyJjdWRhX3N0cmVhbSJdCiAgICAgICAgQiAtLSAibmNjbENvbW1Jbml0UmFuayIgLS0+IEVbIm5jY2xfY29tbSJdCiAgICAgICAgQiAtLSAiY3VkYU1hbGxvYyIgLS0+IEZbImJhcnJpZXJfYnVmZmVyIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJDb2xsZWN0aXZlcyIKICAgICAgICBFIC0tICJuY2NsQWxsUmVkdWNlIiAtLT4gR1siR3JhZGllbnQgQXZlcmFnaW5nIl0KICAgICAgICBFIC0tICJuY2NsQnJvYWRjYXN0IiAtLT4gSFsiV2VpZ2h0IFN5bmMiXQogICAgZW5kCgogICAgQSAtLSAidHJhaW5FcG9jaCIgLS0+IElbIkRpc3RyaWJ1dGVkVHJhaW5lckZ1dGhhcmsiXQogICAgSSAtLSAiYWxsUmVkdWNlKGdyYWRpZW50cykiIC0tPiBC)
---

## Distributed Trainers

JAIDE provides two primary trainer implementations: a standard `DistributedTrainer` for hybrid quantum workloads and a `DistributedTrainerFuthark` optimized for pure GPU performance using Futhark kernels.

### DistributedTrainerFuthark
This trainer focuses on `f16` precision and 100% VRAM-resident training. It utilizes the `RSFAccelerator` to interface with Futhark-generated GPU code.

**Weight-Delta Averaging Pattern:**
Instead of traditional SGD where gradients are averaged, the Futhark trainer often employs a weight-delta pattern:
1.  Local ranks compute updates on their partitioned dataset.
2.  The `allReduce` operation is called with `ncclSum` to aggregate changes.
3.  The result is divided by `world_size` to produce the global update.

### Dataset Partitioning
The trainers handle JSONL datasets by partitioning lines across ranks to ensure each GPU processes unique data.
*   **`loadDataset`**: Opens the JSONL file and extracts usable text lines.
*   **`isUsableDatasetLine`**: Validates lines using a JSON parser and the `MGT` tokenizer to ensure they contain tokenizable content.
*   **Partitioning**: Ranks typically skip lines or load specific ranges based on their `rank` and `world_size` to prevent redundant computation.

### Checkpoint Schema
Checkpoints are serialized using a binary format (Version 4). The schema includes:
*   **Header**: Version and metadata.
*   **Model Dimensions**: `model_dim` and `vocab_size`.
*   **Weights**: Flattened `f32` or `f16` arrays representing the RSF coupling layers.
*   **Optimizer State**: Momentum buffers and global step counts.

Title: Trainer Component Interaction
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJEaXN0cmlidXRlZFRyYWluZXJGdXRoYXJrIgogICAgICAgIERUWyJEaXN0cmlidXRlZFRyYWluZXJGdXRoYXJrIl0KICAgICAgICBUS1siTUdUIFRva2VuaXplciJdCiAgICAgICAgQUNbIlJTRkFjY2VsZXJhdG9yIl0KICAgICAgICBDT1siR1BVQ29vcmRpbmF0b3IiXQogICAgZW5kCgogICAgRFQgLS0+IFRLCiAgICBEVCAtLT4gQUMKICAgIERUIC0tPiBDTwoKICAgIHN1YmdyYXBoICJFeHRlcm5hbCBJL08iCiAgICAgICAgRFNbIi5qc29ubCBEYXRhc2V0Il0KICAgICAgICBDS1siLmNrcHQgQ2hlY2twb2ludCJdCiAgICBlbmQKCiAgICBEVCAtLSAiZXh0cmFjdERhdGFzZXRUZXh0IiAtLT4gRFMKICAgIERUIC0tICJzYXZlQ2hlY2twb2ludCIgLS0+IENL)
---

## Cloud Integration with Modal

The `ModalGPUClient` facilitates deploying these distributed training jobs to cloud infrastructure (e.g., NVIDIA B200/B300 clusters).

### Job Deployment
The client communicates with the Modal API to provision resources and execute the JAIDE container:
*   **GPU Configuration**: Requests specific hardware like "B200" and sets the `gpu_count` (typically 8).
*   **Request Lifecycle**: Uses `std.http.Client` to send POST requests to `/v1/functions/deploy` with a JSON payload containing the `model_path` and `dataset_path`.
*   **Authentication**: Attaches a "Bearer" token to all requests for secure access.

| Function | Purpose |
| :--- | :--- |
| `deployTrainingJob` | Submits a new training task to the Modal cloud |
| `getJobStatus` | Polls the API for the current state of a running job |
| `sendRequest` | Internal helper for handling HTTP compatibility and headers |

## Implementation Details

### Fixed-Point Arithmetic
For specific training configurations, a custom `Fixed32_32` type is used to handle high-precision updates without the overhead of 64-bit floats in certain kernels.

### PRNG
A custom `PRNG` (Pseudo-Random Number Generator) is implemented to ensure reproducible weight initialization across different ranks if they share a seed.

### Shape and Tensor Logic
The distributed system relies on a `Shape` struct that calculates strides and total sizes, ensuring that tensors sent over NCCL are contiguous. The `isContiguous` check is critical before performing `allReduce` operations to prevent memory corruption.


---

*[Back to Table of Contents](#table-of-contents) | Page 22 of 34 | Next: Cloud Training with Modal*

<a id="page-23"></a>

# Cloud Training with Modal




The JAIDE v40 system utilizes **Modal** to provide a scalable, serverless cloud training infrastructure. This environment is specifically optimized for high-performance training on NVIDIA B200 GPUs, leveraging custom-built Docker images that integrate the Zig compiler, Futhark GPU kernel compiler, and the CUDA toolkit. The system supports distributed training across 8×B200 nodes, automated dataset ingestion from HuggingFace, and persistent storage for model checkpoints.

## Image Build Pipeline

The cloud environment is defined by a multi-stage image build process. The image is based on `nvidia/cuda:12.8.1-devel-ubuntu24.04` (for training) or `nvidia/cuda:12.4.0-devel-ubuntu22.04` (for inference) and is provisioned with the specific toolchains required for JAIDE's hybrid architecture.

### Build Stages
1.  **System Dependencies**: Installation of `build-essential`, `git`, `xz-utils`, and `libgomp1`.
2.  **Zig Toolchain**: Installation of Zig 0.13.0, which is required to compile the core JAIDE engine.
3.  **Futhark Compiler**: Integration of the Futhark compiler (nightly or via `opam`) to transform `.fut` kernels into C libraries for GPU acceleration.
4.  **AOT Compilation**: The image attempts to pre-compile the Futhark kernels and the Zig binary during the image build phase to minimize container startup latency.

### Runtime Build Fallback
If the pre-build fails or the source code is modified, the system includes a `_runtime_build` function (or `_runtime_build_inference` for the inference script) that detects missing binaries and recompiles them inside the running container before execution starts.

**Cloud Build and Execution Flow**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJJbWFnZUJ1aWxkIgogICAgICAgIEFbIkJhc2U6IG52aWRpYS9jdWRhIl0gLS0+IEJbIkluc3RhbGwgWmlnIDAuMTMuMCJdCiAgICAgICAgQiAtLT4gQ1siSW5zdGFsbCBGdXRoYXJrIl0KICAgICAgICBDIC0tPiBEWyJBZGQgL2phaWRlX3NyYyJdCiAgICAgICAgRCAtLT4gRVsiZnV0aGFyayBjIC0tbGlicmFyeSJdCiAgICAgICAgRSAtLT4gRlsiemlnIGJ1aWxkIC1Eb3B0aW1pemU9UmVsZWFzZUZhc3QiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlJ1bnRpbWVFeGVjdXRpb24iCiAgICAgICAgR1sibW9kYWxfdHJhaW4ucHkiXSAtLT4gSHsiQmluYXJ5IGV4aXN0cz8ifQogICAgICAgIEggLS0gIk5vIiAtLT4gSVsiX3J1bnRpbWVfYnVpbGQoKSJdCiAgICAgICAgSCAtLSAiWWVzIiAtLT4gSlsiRG93bmxvYWQgRGF0YXNldCJdCiAgICAgICAgSSAtLT4gSgogICAgICAgIEogLS0+IEtbIkRpc3RyaWJ1dGVkIFRyYWluaW5nICg4eEIyMDApIl0KICAgICAgICBLIC0tPiBMWyJTYXZlIE1vZGVsIHRvIC9tb2RlbHMiXQogICAgZW5k)
## Distributed Training Configuration

The training scripts are designed for massive parallelism, specifically targeting an 8×B200 GPU configuration.

### Resource Specification
The `modal_distributed_train.py` script defines high-tier resource requirements to handle the RSF model's memory footprint:
*   **GPU**: `B200:8`.
*   **CPU**: 64.0 Cores (up to 80.0 limit).
*   **Memory**: 256 GB (262144 MB).
*   **Ephemeral Disk**: 3 TB for temporary training artifacts.

### NCCL and GPU Environment
To ensure efficient multi-GPU communication, the script configures the environment for the NVIDIA Collective Communications Library (NCCL). It sets `NCCL_DEBUG=INFO` for troubleshooting and explicitly maps `CUDA_VISIBLE_DEVICES` based on the detected hardware.

## Data and Model Persistence

Modal Volumes are used to provide persistent storage across different cloud runs.

| Volume Name | Mount Path | Purpose |
| :--- | :--- | :--- |
| `jaide-training-data` | `/data` | Stores the processed `finephrase` dataset. |
| `jaide-checkpoints` | `/checkpoints` | Stores intermediate training states and model weights. |

### Dataset Pipeline
The `download_finephrase_to_jsonl` function manages the ingestion of the `HuggingFaceFW/finephrase` dataset. It performs the following steps:
1.  Loads the dataset via the HuggingFace `datasets` library.
2.  Extracts text using prioritized keys: `text`, `content`, `sentence`, or `article`.
3.  Filters for samples longer than 20 characters.
4.  Serializes the result to a `.jsonl` file and commits the volume.

## Training and Inference Entrypoints

The system provides two primary Modal entrypoints: `modal_train.py` for model optimization and `modal_inference.py` for model evaluation.

### Training Logic (`modal_train.py`)
The `train` function is decorated with `@app.function` to specify the 8×B200 GPU requirement. It constructs a command-line execution of the `jaide` binary with the following parameters:
*   `--mode train`
*   `--dataset /dataset/train.jsonl`
*   `--epochs`, `--batch-size`, `--lr`.

Training results, including duration and exit codes, are logged into a `training_history.json` file stored within the persistent volume.

### Inference Logic (`modal_inference.py`)
The `inference` function provides a serverless endpoint for generating text from a trained model. It reloads the `models_volume` to ensure the latest checkpoints are visible, performs a runtime build if necessary, and executes the binary in `--mode infer`.

**Entity Association: Script to Binary Interface**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJNb2RhbFNwYWNlIgogICAgICAgIE1UWyJtb2RhbF90cmFpbi5weSJdCiAgICAgICAgTUlbIm1vZGFsX2luZmVyZW5jZS5weSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZVNwYWNlIgogICAgICAgIEpBWyJtYWluLnppZyAoQmluYXJ5KSJdCiAgICAgICAgRktbImZ1dGhhcmtfa2VybmVscy5mdXQiXQogICAgZW5kCgogICAgTVQgLS0gImV4ZWN1dGVzIC0tbW9kZSB0cmFpbiIgLS0+IEpBCiAgICBNSSAtLSAiZXhlY3V0ZXMgLS1tb2RlIGluZmVyIiAtLT4gSkEKICAgIEpBIC0tICJsaW5rcyIgLS0+IEZLCiAgICAKICAgIHN0eWxlIE1UIHN0cm9rZS1kYXNoYXJyYXk6IDUgNQogICAgc3R5bGUgTUkgc3Ryb2tlLWRhc2hhcnJheTogNSA1)
## Setup and Deployment

The `modal_setup.sh` script automates the initialization of the cloud environment.

1.  **Authentication**: Checks for Modal CLI and runs `modal token new` if required.
2.  **Volume Creation**: Provisions the `jaide-training-data` and `jaide-dataset` volumes.
3.  **Execution Commands**: Provides standard templates for running training with custom hyperparameters, such as `modal run modal_train.py --epochs 100 --dim 1024`.


---

*[Back to Table of Contents](#table-of-contents) | Page 23 of 34 | Next: Hardware Acceleration Layer*

<a id="page-24"></a>

# Hardware Acceleration Layer




The Hardware Acceleration Layer provides the high-performance execution backends for JAIDE v40. It abstracts diverse computational substrates—ranging from GPGPU kernels and CUDA-optimized buffers to FPGA/ASIC Register Transfer Level (RTL) components—into a unified interface used by the neural processing and optimization subsystems.

The layer is divided into three primary domains:
1.  **GPGPU Kernels**: Data-parallel implementations of RSF, SSI, and training operations written in Futhark.
2.  **CUDA Bridge**: Low-level Zig bindings and memory management for NVIDIA hardware.
3.  **RTL Components**: Hardware description logic for custom silicon or FPGA deployment of core search and arbitration tasks.

### Code-to-System Mapping: Acceleration Interfaces

The following diagram illustrates how the high-level Zig abstractions interact with the underlying hardware-specific implementations.

**Hardware Backend Integration Map**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJaaWcgQXBwbGljYXRpb24gU3BhY2UiCiAgICAgICAgQVsiUlNGQWNjZWxlcmF0b3IiXSAtLSAibWFuYWdlcyIgLS0+IEJbIkZ1dGhhcmtDb250ZXh0Il0KICAgICAgICBBIC0tICJ1c2VzIiAtLT4gQ1siUGlubmVkTWVtb3J5Il0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJDb2RlIEVudGl0eSBTcGFjZSIKICAgICAgICBCIC0tICJjYWxscyIgLS0+IERbImZ1dGhhcmtfY29udGV4dF9uZXcoKSJdCiAgICAgICAgQyAtLSAiY2FsbHMiIC0tPiBFWyJjdWRhSG9zdEFsbG9jKCkiXQogICAgICAgIEZbIkZ1dGhhcmtBcnJheTJERjE2Il0gLS0gIndyYXBzIiAtLT4gR1sic3RydWN0X2Z1dGhhcmtfZjE2XzJkIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJIYXJkd2FyZSBEcml2ZXJzIgogICAgICAgIEQgLS0gIk9wZW5DTC9DVURBIiAtLT4gSFsiR1BVIERyaXZlciJdCiAgICAgICAgRSAtLSAiVW5pZmllZCBNZW1vcnkiIC0tPiBICiAgICBlbmQKCiAgICBzdHlsZSBBIHN0cm9rZS13aWR0aDoycHgKICAgIHN0eWxlIEIgc3Ryb2tlLXdpZHRoOjJweAogICAgc3R5bGUgRiBzdHJva2Utd2lkdGg6MnB4)
---

## 7.1 Futhark GPU Kernels
The Futhark library contains the core mathematical kernels for the system, compiled to OpenCL or CUDA. It handles the computationally intensive Reversible Scatter Flow (RSF) forward and backward passes, utilizing butterfly scatter operations and bijective coupling logic.

**Key Capabilities:**
*   **RSF Operations**: `rsf_forward_layer` and `rsf_backward_layer` implement the coupling math (scale/translate) and permutation logic,.
*   **Optimization**: Implements `fisher_diagonal_update` and `spectral_natural_gradient` for the SFD optimizer.
*   **Retrieval**: `topk` and `score_segments` provide accelerated sequence search for the SSI subsystem.

For details, see [Futhark GPU Kernels](#7.1).

---

## 7.2 CUDA Bindings and Accelerator Interface
The Zig-to-CUDA bridge provides the necessary infrastructure to move data between the CPU and GPU with minimal overhead. It utilizes pinned memory for fast DMA transfers and manages the lifecycle of Futhark-allocated device arrays.

**Key Components:**
*   **`FutharkContext`**: Manages the GPU device lifecycle and command synchronization.
*   **`PinnedMemory`**: Wraps `cudaHostAlloc` to provide page-locked memory buffers, essential for high-speed GPU I/O.
*   **Array Wrappers**: Types like `FutharkArray2DF16` provide type-safe handles for multi-dimensional tensors residing in GPU memory.

For details, see [CUDA Bindings and Accelerator Interface](#7.2).

---

## 7.3 Clash RTL Components
For deployment on non-GPGPU hardware (FPGAs or ASICs), JAIDE provides RTL components written in Clash (a functional hardware description language). These components focus on memory arbitration and search acceleration.

**RTL Architecture**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJNZW1vcnlBcmJpdGVyIChNZWFseSBNYWNoaW5lKSIKICAgICAgICBJWyJBcmJJZGxlIl0gLS0gImZpbmRJbmRleCIgLS0+IFNbIkFyYlNlcnZpbmciXQogICAgICAgIFMgLS0gIlNlcnZpY2VDeWNsZXMiIC0tPiBJCiAgICBlbmQKCiAgICBzdWJncmFwaCAiU2lnbmFsIFJvdXRpbmciCiAgICAgICAgUjFbIkNsaWVudCAwIl0gLS0+IE1bIm1lbW9yeUFyYml0ZXIiXQogICAgICAgIFIyWyJDbGllbnQgMSJdIC0tPiBNCiAgICAgICAgTSAtLT4gRFJbIkRSQU0gQ29udHJvbGxlciJdCiAgICAgICAgRFIgLS0gImZpbHRlclJlc3AiIC0tPiBPWyJWYWxpZGF0ZWQgUmVzcG9uc2UiXQogICAgZW5k)
**Key Components:**
*   **`MemoryArbiter`**: A 4-client Mealy state machine that manages concurrent memory access requests, ensuring fair bandwidth distribution via `ServiceCycles`.
*   **`filterResp`**: Logic to route memory responses back to the specific `ClientID4` that initiated the request.

For details, see [Clash RTL Components](#7.3).


---

*[Back to Table of Contents](#table-of-contents) | Page 24 of 34 | Next: Futhark GPU Kernels*

<a id="page-25"></a>

# Futhark GPU Kernels




The Futhark GPU kernel library provides the high-performance acceleration layer for the JAIDE v40 system. It implements the core mathematical operations for the Reversible Scatter Flow (RSF) architecture, the Stochastic Fisher Diagonal (SFD) optimizer, and Structured Sequence Index (SSI) retrieval. These kernels are designed for massive parallelism on CUDA/OpenCL backends, featuring numerical safety patterns (NaN/Inf handling) and fixed-point hardware simulation.

## Core RSF Operations

The RSF architecture relies on bijective coupling layers. The Futhark implementation provides both forward and backward passes, where the backward pass is calculated using the reversible properties of the flow to maintain $O(1)$ memory efficiency relative to depth.

### RSF Forward and Flow
The `rsf_forward` entry point processes batches of input tensors. It splits the input into two halves ($x_1, x_2$) and applies a scale-and-translate transformation:
1.  **Scale**: $y_1 = x_1 \odot \exp(\text{clip}(\text{weights}_s \cdot x_2 + \text{bias}_s))$.
2.  **Translate**: $y_2 = x_2 + (\text{weights}_t \cdot y_1 + \text{bias}_t)$.

The `rsf_scatter` function implements a butterfly Haar-wavelet style mixing, using an `inv_sqrt2` constant (1/√2) to maintain variance during the transformation.

### RSF Backward Pass
The `rsf_backward` kernel computes gradients for weights ($s, t$) and biases. It iterates through the batch, calculating the `dy1_total` by combining the direct gradient from $y_1$ and the backpropagated gradient through the translation function of $y_2$.

| Function | Role | File Reference |
| :--- | :--- | :--- |
| `rsf_forward` | Main entry for forward bijective coupling | |
| `rsf_backward` | Gradient computation for RSF layers | |
| `rsf_scatter` | Butterfly mixing of input dimensions | |
| `rsf_flow` | Internal logic for scale/translate coupling | |

### Data Flow: RSF Coupling Layer
The following diagram illustrates the data flow within the `rsf_flow` kernel, showing the interaction between the split halves and the weights.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJyc2ZfZmxvdyBbc3JjL2h3L2FjY2VsL2Z1dGhhcmtfa2VybmVscy5mdXRdIgogICAgICAgIFhbImlucHV0IFtoYWxmKjJdIl0gLS0+IFgxWyJ4MSBbMDpoYWxmXSJdCiAgICAgICAgWCAtLT4gWDJbIngyIFtoYWxmOmRdIl0KICAgICAgICAKICAgICAgICBYMiAtLT4gTWF0TXVsU1sid2VpZ2h0c19zICogeDIiXQogICAgICAgIE1hdE11bFMgLS0+IEJpYXNTWyIrIHNfYmlhcyJdCiAgICAgICAgQmlhc1MgLS0+IENsaXBTWyJmMzIubWF4IGNsaXBfbWluIl0KICAgICAgICBDbGlwUyAtLT4gRXhwU1siZjMyLmV4cCJdCiAgICAgICAgCiAgICAgICAgWDEgLS0+IFNjYWxlWyIqIHNjYWxlIl0KICAgICAgICBFeHBTIC0tPiBTY2FsZQogICAgICAgIFNjYWxlIC0tPiBZMVsieTEiXQogICAgICAgIAogICAgICAgIFkxIC0tPiBNYXRNdWxUWyJ3ZWlnaHRzX3QgKiB5MSJdCiAgICAgICAgTWF0TXVsVCAtLT4gQmlhc1RbIisgdF9iaWFzIl0KICAgICAgICBCaWFzVCAtLT4gQWRkVFsiKyB4MiJdCiAgICAgICAgQWRkVCAtLT4gWTJbInkyIl0KICAgICAgICAKICAgICAgICBZMSAtLT4gT3V0cHV0WyJ5MSArKyB5MiJdCiAgICAgICAgWTIgLS0+IE91dHB1dAogICAgZW5k)
## SFD Optimizer and Natural Gradient

The Stochastic Fisher Diagonal (SFD) optimizer utilizes second-order information to accelerate convergence. Futhark kernels handle the update of the Fisher information matrix and the application of the natural gradient.

### Fisher Diagonal Update
The `fisher_diagonal_update` function maintains a running estimate of the squared gradients. It includes safety checks for `isnan` and `isinf` to prevent gradient explosion.

### Natural Gradient Application
The `spectral_natural_gradient` function preconditions the gradient by the inverse of the Fisher diagonal. It uses a `damping` factor (defaulting to `1e-8f32`) to ensure numerical stability when the Fisher estimate is near zero.

### Training Step Integration
The `training_step` entry fuses several operations into a single GPU call:
1.  `batch_forward`: Computes predictions.
2.  `batch_compute_loss`: Calculates MSE loss.
3.  `batch_gradients`: Computes all parameter gradients.
4.  `sfd_update_half`: Updates weights using momentum and learning rate.

## Retrieval and SSI Hashing

Retrieval operations for the Structured Sequence Index (SSI) are accelerated via specialized scoring and sorting kernels.

*   **Scoring**: `score_segments` performs parallel hash matching between a `query_hash` and a vector of `segment_hashes`, applying a match bonus to the base scores.
*   **Top-K Selection**: `topk` utilizes a radix sort (imported from `diku-dk/sorts`) to find the highest scoring indices. It uses `f32_total_order` to handle floating-point comparison safely on the GPU.

## Fractal LPU Simulation

The `FractalLPU` and `FractalTile` structures simulate a recursive hardware architecture for processing Non-linear Self-Similar Information Retrieval (NSIR) graphs. This simulation models NoC (Network-on-Chip) routing and core gating.

### Fractal Dimension and Gating
The system uses a `FractalDimensionConfig` defining the `hausdorff_dim` (default 1.5). This controls how `FractalTile` objects subdivide into children.

### Load Balancing and Execution
*   **Load Balancing**: `balanceLoad` redistributes `pending_ops` across `ComputeUnit` arrays if they exceed the `load_balance_factor`.
*   **Fixed-Point Execution**: `executeFixedPoint` simulates hardware arithmetic by scaling inputs by `coherence` (converted to a 16-bit fixed-point `scale`) and performing bit-shifted division.

### System Mapping: Code to Hardware Simulation
The following diagram maps the Zig entities to the simulated hardware components.

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJGcmFjdGFsTFBVIEhhcmR3YXJlIE1vZGVsIFtzcmMvaHcvYWNjZWwvZnJhY3RhbF9scHUuemlnXSIKICAgICAgICBMUFVbIkZyYWN0YWxMUFUiXSAtLT4gUm9vdFsicm9vdF90aWxlOiBGcmFjdGFsVGlsZSJdCiAgICAgICAgUm9vdCAtLT4gQ2hpbGRyZW5bImNoaWxkcmVuOiBbXT8qRnJhY3RhbFRpbGUiXQogICAgICAgIFJvb3QgLS0+IENVc1siY29tcHV0ZV91bml0czogW11Db21wdXRlVW5pdCJdCiAgICAgICAgCiAgICAgICAgQ1VzIC0tPiBPcHNbInBlbmRpbmdfb3BzOiB1NjQiXQogICAgICAgIENVcyAtLT4gQWRkclsiYmFzZV9hZGRyOiB1NjQiXQogICAgICAgIAogICAgICAgIHN1YmdyYXBoICJTaW11bGF0aW9uIExvZ2ljIgogICAgICAgICAgICBCQUxbImJhbGFuY2VMb2FkKCkiXQogICAgICAgICAgICBGUFsiZXhlY3V0ZUZpeGVkUG9pbnQoKSJdCiAgICAgICAgICAgIE1BUFsibWFwU1NSR05vZGUoKSJdCiAgICAgICAgZW5kCiAgICBlbmQKICAgIAogICAgU1NSR1siU1NSRyBOb2RlIEhhc2giXSAtLSAibm9kZV9oYXNoICUgbGVuIiAtLT4gTUFQCiAgICBNQVAgLS0gImluY3JlbWVudCIgLS0+IE9wcwogICAgQkFMIC0tICJjbGFtcCB0byBhdmcgKiBmYWN0b3IiIC0tPiBPcHMKICAgIEZQIC0tICJpbnB1dCAqIHNjYWxlID4+IDE2IiAtLT4gT3V0cHV0WyJGaXhlZC1Qb2ludCBPdXRwdXQiXQ==)
## Zig Bindings and Context Management

The `futhark_bindings.zig` file provides the FFI layer between the Zig runtime and the compiled Futhark C code.

*   **Context Management**: `futhark_context_new` and `futhark_context_config_set_device` allow the Zig side to initialize the GPU environment.
*   **Memory Interop**: Opaque pointers like `struct_futhark_f16_2d` represent GPU-resident arrays. Functions like `futhark_new_f16_2d` upload data, while `futhark_values_f16_2d` download results.
*   **Entry Points**: Every Futhark `entry` function is exposed as a `futhark_entry_*` C function, such as `futhark_entry_rsf_forward`.


---

*[Back to Table of Contents](#table-of-contents) | Page 25 of 34 | Next: CUDA Bindings and Accelerator Interface*

<a id="page-26"></a>

# CUDA Bindings and Accelerator Interface




The CUDA Bindings and Accelerator Interface provide the low-level bridge between the Zig-based RSF neural core and GPU hardware. This subsystem abstracts memory management (pinned host memory vs. device memory), handles version synchronization between CPU and GPU buffers, and provides the `RSFAccelerator` interface for executing high-performance kernels.

## System Architecture

The acceleration layer is structured into three distinct levels: the raw C foreign function interface (FFI) for CUDA, the Futhark-generated kernel bindings, and the high-level Zig `RSFAccelerator` which orchestrates data flow between the two.

### Code Entity Relationship
This diagram maps the natural language components to their specific code entities and files.

"Entity Mapping: Acceleration Subsystem"
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJaaWcgU3BhY2UiCiAgICAgICAgQVsiUlNGQWNjZWxlcmF0b3IiXSAtLSAidXNlcyIgLS0+IEJbIkZ1dGhhcmtDb250ZXh0Il0KICAgICAgICBBIC0tICJtYW5hZ2VzIiAtLT4gQ1siUGlubmVkTWVtb3J5Il0KICAgICAgICBEWyJhY2NlbF9pbnRlcmZhY2UuemlnIl0gLS0gImV4cG9ydHMiIC0tPiBBCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQmluZGluZyBTcGFjZSIKICAgICAgICBCIC0tICJ3cmFwcyIgLS0+IEVbImZ1dGhhcmtfYmluZGluZ3MuemlnIl0KICAgICAgICBDIC0tICJjYWxscyIgLS0+IEZbImN1ZGFIb3N0QWxsb2MiXQogICAgICAgIEdbImN1ZGFfYmluZGluZ3MuemlnIl0gLS0gImRlZmluZXMiIC0tPiBGCiAgICBlbmQKCiAgICBzdWJncmFwaCAiSGFyZHdhcmUgU3BhY2UiCiAgICAgICAgRSAtLSAiZXhlY3V0ZXMgb24iIC0tPiBIWyJOVklESUEgR1BVIl0KICAgICAgICBGIC0tICJhbGxvY2F0ZXMgb24iIC0tPiBJWyJIb3N0IFJBTSAoUGlubmVkKSJdCiAgICBlbmQ=)
## CUDA Bindings (`cuda_bindings.zig`)

The `cuda_bindings.zig` file provides a type-safe Zig wrapper around the CUDA Driver and Runtime APIs. It handles error translation from `cudaError_t` enums to Zig errors.

### Key Functions
- `cudaHostAlloc`: Allocates page-locked (pinned) host memory that is accessible to the GPU, enabling high-speed DMA transfers.
- `cudaMemcpy` / `cudaMemcpyAsync`: Synchronous and asynchronous data transfer between Host and Device.
- `toError`: A utility function that converts CUDA return codes into the `CudaError` error set.

## Accelerator Interface (`accel_interface.zig`)

This module provides the `RSFAccelerator` and associated memory abstractions. It uses conditional compilation via the `gpu_acceleration` build option to determine if GPU code paths should be active.

### Pinned Memory Management
The `PinnedMemory` struct ensures that neural network weights stored in RAM are "pinned," preventing the OS from swapping them to disk and allowing the GPU to access them via direct memory access (DMA).

- `alloc(size)`: Calls `cudaHostAlloc` with `cudaHostAllocDefault`.
- `asSlice(T)`: Casts the raw pointer to a Zig slice for standard array access.

### Futhark Integration
Futhark is used to generate the high-performance GPU kernels for RSF operations. The `FutharkContext` struct manages the lifecycle of the GPU device context.
- `init()`: Configures device 0, sets group sizes (256), and tile sizes (32) before initializing the context.
- `FutharkArray2DF16`: A wrapper for `f16` (half-precision) 2D arrays used for model weights and activations.

## RSFAccelerator and Data Flow

The `RSFAccelerator` (defined in the RSF module but utilizing the `accel_interface`) manages the synchronization of `f16` weight buffers. It is responsible for the `forwardFromTensor` operation, which moves data from the CPU `Tensor` primitive to the GPU for processing.

### Execution Pipeline
The following diagram illustrates the data flow from a standard CPU `Tensor` through the `RSFAccelerator` to the GPU kernels.

"Data Flow: CPU Tensor to GPU Execution"
![Diagram](https://mermaid.ink/img/c2VxdWVuY2VEaWFncmFtCiAgICBwYXJ0aWNpcGFudCBUIGFzIGNvcmVfdGVuc29yLlRlbnNvcgogICAgcGFydGljaXBhbnQgQSBhcyBSU0ZBY2NlbGVyYXRvcgogICAgcGFydGljaXBhbnQgUCBhcyBQaW5uZWRNZW1vcnkKICAgIHBhcnRpY2lwYW50IEYgYXMgRnV0aGFya0FycmF5MkRGMTYKICAgIHBhcnRpY2lwYW50IEsgYXMgR1BVIEtlcm5lbAoKICAgIFQtPj5BOiBmb3J3YXJkRnJvbVRlbnNvcihpbnB1dCkKICAgIEEtPj5QOiBNYXAgaW5wdXQgdG8gUGlubmVkTWVtb3J5CiAgICBQLT4+RjogQ3JlYXRlIEZ1dGhhcmtBcnJheSAoSG9zdC10by1EZXZpY2UpCiAgICBOb3RlIG92ZXIgRixLOiBLZXJuZWwgRXhlY3V0aW9uIChmMTYpCiAgICBLLT4+RjogT3V0cHV0IFJlc3VsdHMKICAgIEYtPj5BOiB2YWx1ZXMyRCgpIChEZXZpY2UtdG8tSG9zdCkKICAgIEEtPj5UOiBVcGRhdGUgVGVuc29yIFJlc3VsdA==)
### Version Synchronization
The accelerator maintains a version counter for weight buffers. When weights are updated on the CPU (e.g., during an optimizer step), the `RSFAccelerator` detects the version mismatch and triggers a `cudaMemcpy` to sync the GPU's `f16` buffers before the next forward pass.

## Error Handling

The interface defines a comprehensive error set `AccelError` to handle hardware-specific failures:
- `FutharkSyncFailed`: Occurs if the GPU context fails to synchronize after a kernel launch.
- `CudaHostAllocFailed`: Occurs if the driver cannot allocate pinned memory, often due to system memory pressure.
- `InvalidDimensions`: Raised when the input `Tensor` shape does not match the allocated `FutharkArray`.


---

*[Back to Table of Contents](#table-of-contents) | Page 26 of 34 | Next: Clash RTL Components*

<a id="page-27"></a>

# Clash RTL Components




This page documents the hardware-level Register Transfer Level (RTL) components implemented in Clash (Haskell-to-RTL). These components provide high-performance, synthesis-ready hardware cores for memory arbitration, sequence indexing, and ranking, targeting ASIC or FPGA deployment.

## Overview of Clash RTL Architecture

The hardware components are designed using a synchronous, type-safe approach provided by Clash. The system utilizes Mealy state machines to manage complex control logic, such as tree traversal and multi-client resource contention.

### Hardware-to-Code Mapping

The following diagram bridges the functional hardware requirements to the specific Haskell entities and data types defined in the RTL source.

**Hardware Entity Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJNZW1vcnkgU3Vic3lzdGVtIgogICAgICAgIE1BWyJNZW1vcnlBcmJpdGVyIl0gLS0+IE1SUVsiTWVtUmVxdWVzdCBbc3JjL2h3L3J0bC9NZW1vcnlBcmJpdGVyLmhzXSJdCiAgICAgICAgTUEgLS0+IE1SU1siTWVtUmVzcG9uc2UgW3NyYy9ody9ydGwvTWVtb3J5QXJiaXRlci5oc10iXQogICAgICAgIE1BIC0tPiBBU1RbIkFyYml0ZXJTdGF0ZSBbc3JjL2h3L3J0bC9NZW1vcnlBcmJpdGVyLmhzXSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiU2VhcmNoICYgUmV0cmlldmFsIgogICAgICAgIFNTSVsiU1NJU2VhcmNoIl0gLS0+IFNTVFsiU2VhcmNoU3RhdGUgW3NyYy9ody9ydGwvU1NJU2VhcmNoLmhzXSJdCiAgICAgICAgU1NJIC0tPiBUTlsiVHJlZU5vZGUgW3NyYy9ody9ydGwvU1NJU2VhcmNoLmhzXSJdCiAgICAgICAgUkNbIlJhbmtlckNvcmUiXSAtLT4gUlNUWyJSYW5rZXJTdGF0ZSBbc3JjL2h3L3J0bC9SYW5rZXJDb3JlLmhzXSJdCiAgICAgICAgUkMgLS0+IFJSUVsiUmFua1JlcXVlc3QgW3NyYy9ody9ydGwvUmFua2VyQ29yZS5oc10iXQogICAgZW5kCgogICAgTUEgLS0gIk1lbW9yeSBBY2Nlc3MiIC0tPiBTU0kKICAgIFNTSSAtLSAiQ2FuZGlkYXRlcyIgLS0+IFJD)
---

## MemoryArbiter

The `MemoryArbiter` manages access to a shared memory resource for up to 4 concurrent clients (`NumClients` = 4). It implements a fair-access policy using a Mealy state machine that cycles through requests.

### State Machine and Arbitration Logic
The arbiter operates in two primary states:
1.  **ArbIdle**: The arbiter searches for the first available request from the client vector using `findIndex isJust`.
2.  **ArbServing**: Once a client is granted access, the arbiter enters a serving state for a fixed duration defined by `ServiceCycles` (default 4).

### Response Routing
Responses from memory are broadcast to all clients but filtered by the `filterResp` function. This ensures that a client only receives a `MemResponse` if the `respClient` ID matches its own `ClientID4`.

| Component | Type | Description |
| :--- | :--- | :--- |
| `Addr32` | `Unsigned 32` | 32-bit memory address. |
| `Data64` | `Unsigned 64` | 64-bit data word. |
| `ClientID4` | `Unsigned 4` | Identifier for one of the 4 clients. |

---

## SSISearch Core

The `SSISearch` component implements hardware-accelerated traversal of the Structured Sequence Index (SSI) tree. It is designed to resolve `SearchRequest` queries by traversing `TreeNode` structures stored in memory.

### Search Logic Flow
The search process is a 3-state cycle:
1.  **Idle**: Waiting for a `SearchRequest`.
2.  **Fetching**: Issuing a memory request for a specific `NodeAddr32`.
3.  **Comparing**: Comparing the `searchKey` against the `nodeKey`. Depending on the result, it either terminates (found/not found) or moves to the `leftChild` or `rightChild`.

### Constraints and Safety
*   **Max Depth**: The search is bounded by `MaxSearchDepthConfig` (64) to prevent infinite loops in malformed trees.
*   **Null Pointers**: The core recognizes `NodeAddr32 0` as a `nullAddr`, indicating a leaf node termination.

**Search State Transitions**
![Diagram](https://mermaid.ink/img/c3RhdGVEaWFncmFtLXYyCiAgICBbKl0gLS0+IElkbGUKICAgIElkbGUgLS0+IEZldGNoaW5nIDogU2VhcmNoUmVxdWVzdCBSZWNlaXZlZAogICAgRmV0Y2hpbmcgLS0+IENvbXBhcmluZyA6IFRyZWVOb2RlIExvYWRlZAogICAgQ29tcGFyaW5nIC0tPiBGZXRjaGluZyA6IEtleSBNaXNtYXRjaCAoR28gTGVmdC9SaWdodCkKICAgIENvbXBhcmluZyAtLT4gSWRsZSA6IEtleSBNYXRjaCAoRm91bmQpCiAgICBDb21wYXJpbmcgLS0+IElkbGUgOiBMZWFmIFJlYWNoZWQgKE5vdCBGb3VuZCkKICAgIEZldGNoaW5nIC0tPiBJZGxlIDogRGVwdGggRXhjZWVkZWQ=)
---

## RankerCore

The `RankerCore` provides a pipelined scoring engine for ranking retrieved segments. It combines a `baseScore` with a calculated `positionBias` to produce a `finalScore`.

### Scoring Formula
The hardware implements a reciprocal position bias to penalize segments appearing later in a sequence:
*   **Position Bias**: `positionBiasScale / (segmentPos + 1)`.
*   **Final Score**: `baseScore + positionBias`.

### Rank Tracking
The `RankerState` tracks the `lastQuery` hash. If subsequent `RankRequest` objects share the same query hash, the `stateCounter` increments, effectively assigning a rank index to each result in a stream.

### Data Structures
*   **RankRequest**: Contains `queryHash`, `segmentID`, `segmentPos`, and `baseScore`.
*   **RankResult**: Outputs the `resultID`, the `finalScore`, and the calculated `rank`.

---

## Synthesis Targets

All components define a `topEntity` which exposes the necessary `Clock`, `Reset`, and `Enable` signals for standard FPGA/ASIC synthesis tools (e.g., Vivado, Quartus, or Yosys).

*   **MemoryArbiter Top**:
*   **SSISearch Top**:
*   **RankerCore Top**:

---

*[Back to Table of Contents](#table-of-contents) | Page 27 of 34 | Next: Inference Server and API*

<a id="page-28"></a>

# Inference Server and API




The JAIDE v40 Inference Server provides a high-performance HTTP/1.1 interface for interacting with the neural core and relational graph systems. It handles the full lifecycle of a request—from raw text ingestion and tokenization to reversible flow processing and NSIR-guided modulation.

### System Overview

The server is built around the `InferenceServer` class, which manages a pool of worker threads to handle concurrent TCP connections. It utilizes a custom `RateLimiter` to enforce request quotas and provides a RESTful API surface for health monitoring and model inference.

#### Inference Server Architecture
The following diagram illustrates the relationship between the HTTP server components and the underlying processing engine.

"Inference Server Components"
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJIVFRQX0xheWVyIgogICAgICAgIEFbIkluZmVyZW5jZVNlcnZlciJdIC0tICJtYW5hZ2VzIiAtLT4gQlsiU2VydmVyQ29uZmlnIl0KICAgICAgICBBIC0tICJ1c2VzIiAtLT4gQ1siUmF0ZUxpbWl0ZXIiXQogICAgICAgIEEgLS0gImxpc3RlbnMiIC0tPiBEWyJzdGQubmV0LlN0cmVhbVNlcnZlciJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiUmVxdWVzdF9Db250ZXh0IgogICAgICAgIEVbIkluZmVyZW5jZVJlcXVlc3QiXSAtLT4gRlsiTUdUIFRva2VuaXplciJdCiAgICAgICAgRiAtLT4gR1siUlNGTGF5ZXIgUHJvY2Vzc2luZyJdCiAgICAgICAgRyAtLT4gSFsiTlNJUiBNb2R1bGF0aW9uIl0KICAgICAgICBIIC0tPiBJWyJJbmZlcmVuY2VSZXNwb25zZSJdCiAgICBlbmQKCiAgICBEIC0tICJzcGF3bnMiIC0tPiBFCiAgICBBIC0tICJyZWZlcmVuY2VzIiAtLT4gRwogICAgQSAtLSAicmVmZXJlbmNlcyIgLS0+IEg=)
---

### API Surface

The server exposes two primary endpoints via standard HTTP/1.1.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/health` | `GET` | Returns server status, uptime, and model loading state. |
| `/v1/inference` | `POST` | Processes text through the RSF/NSIR pipeline. |

#### Request and Response Schemas
Requests are submitted as JSON objects. The `InferenceRequest` struct defines the expected fields, including the input `text` and optional `max_tokens`. Responses are returned as `InferenceResponse` objects containing generated `tokens`, optional `embeddings`, and high-precision `processing_time_ms`.

---

### Server Configuration and Execution

The server is configured via the `ServerConfig` struct, which controls networking parameters (port, host, max connections), security (API key requirements), and performance (batch size, rate limits).

The `jaide-inference-server` executable parses command-line arguments to override these defaults.

#### Code-to-System Mapping
This diagram bridges the CLI configuration logic to the internal server state.

"CLI to Server Initialization"
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJDTElfRW50cnlwb2ludCIKICAgICAgICBNQUlOWyJzcmMvaW5mZXJlbmNlX3NlcnZlcl9tYWluLnppZyJdIC0tICJwYXJzZXMiIC0tPiBBUkdTWyJzdGQucHJvY2Vzcy5hcmdzIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJDb25maWd1cmF0aW9uX0xvZ2ljIgogICAgICAgIEFSR1MgLS0gInBvcHVsYXRlcyIgLS0+IENGR1siU2VydmVyQ29uZmlnIl0KICAgICAgICBDRkcgLS0gInBvcnQvaG9zdCIgLS0+IFNSVlsiSW5mZXJlbmNlU2VydmVyLmluaXQiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIlNlcnZlcl9TdGF0ZSIKICAgICAgICBTUlYgLS0gImFsbG9jYXRlcyIgLS0+IEFMQ1siR2VuZXJhbFB1cnBvc2VBbGxvY2F0b3IiXQogICAgICAgIFNSViAtLSAibG9hZHMiIC0tPiBNRExbIk1vZGVsRm9ybWF0LmltcG9ydE1vZGVsIl0KICAgIGVuZA==)
---

### Detailed Implementation Modules

The inference server's responsibilities are split between the request lifecycle management and the verified execution engine.

#### [InferenceServer — HTTP API and Request Lifecycle](#8.1)
This child page details the internal mechanics of `InferenceServer`. It covers:
* **Request Lifecycle**: Manual HTTP/1.1 header parsing and `InferenceRequest` validation.
* **Processing Pipeline**: The sequence from `MGT` tokenization to `RSFLayer` embedding and `SSI` indexing.
* **Post-Processing**: How `nsirModulateForInference` adjusts outputs based on the relational graph state.
* **Memory Strategy**: Use of per-request `ArenaAllocator` to ensure zero-leak high-concurrency performance.

#### [Verified Inference Engine and ZK Proofs](#8.2)
This child page covers the `VerifiedInferenceEngine`, which provides cryptographic guarantees for the model's output. It covers:
* **ZK Proofs**: Integration with `ZKInferenceProver` and the `Circom` inference trace.
* **Privacy**: Application of Laplace noise for Differential Privacy.
* **Integrity**: `Blake3` commitment schemes and `BatchVerifier` for rolling hash validation across request batches.

---

---

*[Back to Table of Contents](#table-of-contents) | Page 28 of 34 | Next: InferenceServer — HTTP API and Request Lifecycle*

<a id="page-29"></a>

# InferenceServer — HTTP API and Request Lifecycle




The `InferenceServer` is the primary interface for external consumers to interact with the JAIDE v40 system. It provides a high-performance HTTP/1.1 API that orchestrates the transition from raw text input to neural embeddings and indexed retrieval. The server is designed for high concurrency using a multi-threaded architecture, per-request arena memory management, and a custom rolling-window rate limiter.

### Server Configuration and Initialization

The server is configured via the `ServerConfig` struct, which defines networking parameters, security requirements, and model paths.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `port` | `u16` | `8080` | Port for the TCP listener. |
| `host` | `[]const u8` | `"127.0.0.1"` | Bind address. |
| `max_connections` | `u32` | `100` | Maximum concurrent TCP connections. |
| `rate_limit_per_minute` | `u32` | `10` | Requests allowed per IP in a 60s window. |
| `require_api_key` | `bool` | `true` | Whether to validate `X-API-Key` headers. |
| `max_request_size_bytes`| `usize` | `1MB` | Safety limit for incoming HTTP bodies. |

### HTTP API Endpoints

The server implements a manual HTTP/1.1 parser to minimize overhead and avoid external dependencies.

#### 1. GET /v1/health
Returns the current status of the server, uptime, and whether the model weights are successfully loaded into memory.
*   **Response Schema**: `HealthResponse`
*   **Fields**: `status`, `uptime_seconds`, `model_loaded`, `version`.

#### 2. POST /v1/inference
The primary entry point for text processing. It accepts a JSON payload and returns token IDs and optional embeddings.
*   **Request Schema**: `InferenceRequest`
*   **Response Schema**: `InferenceResponse`

### Request Lifecycle and Data Flow

The lifecycle of an inference request involves several stages, moving from the network layer through the neural pipeline and back.

#### Pipeline Architecture
The following diagram illustrates the transition from the HTTP request to the internal processing entities.

**Inference Request Pipeline**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOZXR3b3JrIExheWVyIgogICAgICAgIFRDUFsic3RkLm5ldC5TdHJlYW1TZXJ2ZXIiXSAtLT4gUEFSU0VbIk1hbnVhbCBIVFRQIFBhcnNlciJdCiAgICAgICAgUEFSU0UgLS0+IExJTUlUWyJSYXRlTGltaXRlciJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiUHJvY2Vzc2luZyBQaXBlbGluZSAoQXJlbmEgTWVtb3J5KSIKICAgICAgICBMSU1JVCAtLT4gVE9LWyJNR1QgVG9rZW5pemVyIl0KICAgICAgICBUT0sgLS0+IEVNQlsiUlNGIEVtYmVkZGluZyJdCiAgICAgICAgRU1CIC0tPiBTU0lbIlNTSSBJbmRleGluZyJdCiAgICAgICAgU1NJIC0tPiBNT0RbIm5zaXJNb2R1bGF0ZUZvckluZmVyZW5jZSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiUmVzcG9uc2UgR2VuZXJhdGlvbiIKICAgICAgICBNT0QgLS0+IEpTT05bIkluZmVyZW5jZVJlc3BvbnNlLnRvSnNvbiJdCiAgICAgICAgSlNPTiAtLT4gU0VORFsiVENQIFN0cmVhbSBXcml0ZSJdCiAgICBlbmQKCiAgICBzdHlsZSBUQ1Agc3Ryb2tlLWRhc2hhcnJheTogNSA1CiAgICBzdHlsZSBUT0sgc3Ryb2tlLXdpZHRoOjJweAogICAgc3R5bGUgRU1CIHN0cm9rZS13aWR0aDoycHgKICAgIHN0eWxlIFNTSSBzdHJva2Utd2lkdGg6MnB4)
### Implementation Details

#### Rate Limiting
The `RateLimiter` uses a rolling 60-second window implemented via a `StringHashMap` of `RequestLog` structs. Each log tracks timestamps of recent requests from a specific IP address.
*   **Mechanism**: When a request arrives, `checkAndRecord` is called. It prunes timestamps older than 60 seconds and checks if the remaining count exceeds `max_requests`.
*   **Concurrency**: Thread safety is maintained via a global mutex on the `RateLimiter` and individual mutexes on each `RequestLog`.

#### Memory Management
To ensure high performance and prevent leaks, the server utilizes a per-request `ArenaAllocator`.
*   Each incoming connection spawns a thread (or uses a pool) where an arena is initialized.
*   All intermediate structures—`InferenceRequest` parsing, `MGT` token buffers, and `InferenceResponse` strings—are allocated within this arena.
*   The entire arena is deallocated once the HTTP response is sent, providing O(1) cleanup.

#### Processing Logic: Tokenization to Modulation
1.  **Tokenization**: The `MGT` (Morpheme-Guided Tokenizer) converts input text into a sequence of token IDs.
2.  **Embedding**: Tokens are passed through the `RSFLayer` to generate high-dimensional vectors.
3.  **SSI Indexing**: The vectors are queried against the `SSI` (Structured Sequence Index) to find relevant historical context or knowledge anchors.
4.  **Post-Processing**: The `nsirModulateForInference` function (from `nsir_core.zig`) adjusts the output embeddings based on the current state of the quantum-relational graph, ensuring the response is contextually grounded in the NSIR knowledge base.

### Error Handling

The server uses standard HTTP status codes to communicate failure states:

| Code | Scenario | Code Entity |
| :--- | :--- | :--- |
| `400 Bad Request` | Invalid JSON or missing "text" field. | `error.InvalidJson`, `error.MissingTextField` |
| `401 Unauthorized` | Missing or invalid `X-API-Key` (if required). | `ServerConfig.require_api_key` |
| `429 Too Many Requests` | Rate limit exceeded. | `RateLimiter.checkAndRecord` |
| `500 Internal Error` | Model not loaded or allocation failure. | `error.OutOfMemory`, `model_loaded == false` |

### System Entity Mapping

This diagram bridges the conceptual "Inference" process with the specific Zig source files and structs responsible for each stage.

**Code Entity Mapping**
![Diagram](https://mermaid.ink/img/Y2xhc3NEaWFncmFtCiAgICBjbGFzcyBJbmZlcmVuY2VTZXJ2ZXIgewogICAgICAgICtTZXJ2ZXJDb25maWcgY29uZmlnCiAgICAgICAgK1JhdGVMaW1pdGVyIHJhdGVfbGltaXRlcgogICAgICAgICtzdGFydCgpCiAgICAgICAgK2hhbmRsZUNvbm5lY3Rpb24oKQogICAgfQogICAgY2xhc3MgTUdUIHsKICAgICAgICArZW5jb2RlKHRleHQpCiAgICB9CiAgICBjbGFzcyBSU0ZMYXllciB7CiAgICAgICAgK2ZvcndhcmRJblBsYWNlKHRlbnNvcikKICAgIH0KICAgIGNsYXNzIFNTSSB7CiAgICAgICAgK3JldHJpZXZlVG9wSyh2ZWN0b3IpCiAgICB9CiAgICBjbGFzcyBOU0lSX0NvcmUgewogICAgICAgICtuc2lyTW9kdWxhdGVGb3JJbmZlcmVuY2UoKQogICAgfQoKICAgIEluZmVyZW5jZVNlcnZlciAuLj4gTUdUIDogIlVzZXMgZm9yIHRva2VuaXphdGlvbiIKICAgIEluZmVyZW5jZVNlcnZlciAuLj4gUlNGTGF5ZXIgOiAiVXNlcyBmb3IgZW1iZWRkaW5nIgogICAgSW5mZXJlbmNlU2VydmVyIC4uPiBTU0kgOiAiVXNlcyBmb3IgcmV0cmlldmFsIgogICAgSW5mZXJlbmNlU2VydmVyIC4uPiBOU0lSX0NvcmUgOiAiVXNlcyBmb3IgbW9kdWxhdGlvbiIKCiAgICBub3RlIGZvciBJbmZlcmVuY2VTZXJ2ZXIgIkRlZmluZWQgaW4gc3JjL2FwaS9pbmZlcmVuY2Vfc2VydmVyLnppZyIKICAgIG5vdGUgZm9yIE1HVCAiRGVmaW5lZCBpbiBzcmMvdG9rZW5pemVyL21ndC56aWciCiAgICBub3RlIGZvciBSU0ZMYXllciAiRGVmaW5lZCBpbiBzcmMvcHJvY2Vzc29yL3JzZi56aWciCiAgICBub3RlIGZvciBTU0kgIkRlZmluZWQgaW4gc3JjL2luZGV4L3NzaS56aWci)

---

*[Back to Table of Contents](#table-of-contents) | Page 29 of 34 | Next: Verified Inference Engine and ZK Proofs*

<a id="page-30"></a>

# Verified Inference Engine and ZK Proofs




The `VerifiedInferenceEngine` provides a cryptographically secure execution environment for JAIDE v40's neural inference. It integrates Zero-Knowledge (ZK) proofs, Differential Privacy (DP), and commitment schemes to ensure that model outputs are both correct (mathematically proven to originate from the specific model weights) and private (protected against data leakage via noise injection).

## Architecture and Data Flow

The engine orchestrates several sub-components to generate a `ZKInferenceProof`. It manages the lifecycle of model weights, execution traces, and the underlying ZK prover.

### System Interaction Diagram

This diagram illustrates the relationship between the high-level `VerifiedInferenceEngine` and the cryptographic primitives defined in `zk_verification.zig` and `dataset_obfuscation.zig`.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJWZXJpZmllZCBJbmZlcmVuY2UgRW5naW5lIFNwYWNlIgogICAgICAgIFZJRVsiVmVyaWZpZWRJbmZlcmVuY2VFbmdpbmUgW3NyYy9jb3JlX3JlbGF0aW9uYWwvdmVyaWZpZWRfaW5mZXJlbmNlX2VuZ2luZS56aWddIl0KICAgICAgICBaS1BbIlpLSW5mZXJlbmNlUHJvdmVyIFtzcmMvY29yZV9yZWxhdGlvbmFsL3prX3ZlcmlmaWNhdGlvbi56aWddIl0KICAgICAgICBEUFsiRGlmZmVyZW50aWFsUHJpdmFjeSBbc3JjL2NvcmVfcmVsYXRpb25hbC96a192ZXJpZmljYXRpb24uemlnXSJdCiAgICAgICAgQ1NbIkNvbW1pdG1lbnRTY2hlbWUgW3NyYy9jb3JlX3JlbGF0aW9uYWwvemtfdmVyaWZpY2F0aW9uLnppZ10iXQogICAgZW5kCgogICAgc3ViZ3JhcGggIk9iZnVzY2F0aW9uICYgU2FmZXR5IFNwYWNlIgogICAgICAgIFBPQ1siUHJvb2ZPZkNvcnJlY3RuZXNzIFtzcmMvY29yZV9yZWxhdGlvbmFsL2RhdGFzZXRfb2JmdXNjYXRpb24uemlnXSJdCiAgICAgICAgREZbIkRhdGFzZXRGaW5nZXJwcmludCBbc3JjL2NvcmVfcmVsYXRpb25hbC9kYXRhc2V0X29iZnVzY2F0aW9uLnppZ10iXQogICAgICAgIEhFWyJIb21vbW9ycGhpY0VuY3J5cHRpb24gW3NyYy9jb3JlX3JlbGF0aW9uYWwvZGF0YXNldF9vYmZ1c2NhdGlvbi56aWddIl0KICAgIGVuZAoKICAgIFZJRSAtLT58b3duc3wgWktQCiAgICBWSUUgLS0+fHVzZXN8IERQCiAgICBWSUUgLS0+fHVzZXN8IENTCiAgICBWSUUgLS0+fG1hbmFnZXN8IFBPQwogICAgVklFIC0tPnx2YWxpZGF0ZXN8IERGCiAgICBWSUUgLS0+fHdyYXBzfCBIRQoKICAgIFpLUCAtLT58ZXhlY3V0ZXN8IENJUlsiaW5mZXJlbmNlX3RyYWNlLmNpcmNvbSBbc3JjL3prL2luZmVyZW5jZV90cmFjZS5jaXJjb21dIl0KICAgIFBPQyAtLT58Z2VuZXJhdGVzfCBUUkFDRVsiRXhlY3V0aW9uIFRyYWNlIl0KICAgIENTIC0tPnxoYXNoZXN8IEJMQUtFM1siQmxha2UzIFtzdGQuY3J5cHRvLmhhc2guQmxha2UzXSJd)
---

## Core Components

### VerifiedInferenceEngine
The central coordinator for secure inference. It can be initialized with or without ZK support using `init` or `initWithZKProofs`.

*   **Weight Management**: Loads and stores layer weights for `s` (scale) and `t` (translate) functions in the RSF architecture.
*   **Verification Tracking**: Maintains `verification_count` and `successful_verifications` to monitor engine integrity.
*   **Model Integrity**: Uses a `model_hash` (Blake3) derived from a constant seed to ensure the model architecture has not been tampered with.

### ZKInferenceProver
The bridge to the SNARK (Groth16) backend. It handles the transformation of floating-point inference operations into circuit-compatible fixed-point witnesses.

*   **Circom Integration**: Compiles `.circom` circuits and manages keys via `CircomProver`.
*   **Witness Generation**: Converts `f32` tensors into fixed-point integers using `precision_bits` (default 64) for circuit consumption.
*   **Batch Verification**: Utilizes a `BatchVerifier` with a rolling hash to validate multiple proofs efficiently.

### Differential Privacy (DP)
To prevent membership inference attacks, the engine injects Laplace noise into the inference results.
*   **Laplace Noise**: Implemented in `DifferentialPrivacy.applyLaplaceNoise`.
*   **Privacy Budget**: Managed via `epsilon`, `delta`, and `sensitivity` parameters.

---

## ZK Circuit: inference_trace.circom

The core logic of the verified inference is defined in Circom, targeting the Groth16 proof system. It performs fixed-point arithmetic to simulate the RSF (Reversible Scatter Flow) computation.

### Logic Flow in Circuit
The circuit validates that for a given input $x$, the output $y$ was correctly computed using the committed model weights.

![Diagram](https://mermaid.ink/img/c2VxdWVuY2VEaWFncmFtCiAgICBwYXJ0aWNpcGFudCBQIGFzIFByb3ZlciAoWmlnKQogICAgcGFydGljaXBhbnQgQyBhcyBDaXJjb20gQ2lyY3VpdAogICAgcGFydGljaXBhbnQgViBhcyBWZXJpZmllcgoKICAgIFAtPj5DOiBJbnB1dCBUZW5zb3JzIChGaXhlZC1Qb2ludCkKICAgIFAtPj5DOiBMYXllciBXZWlnaHRzIChzLCB0KQogICAgTm90ZSBvdmVyIEM6IFJTRkxheWVyQ29tcHV0YXRpb24KICAgIEMtPj5DOiBUYXlsb3IgZXhwYW5zaW9uIGZvciBTY2FsZSAoZXhwKQogICAgQy0+PkM6IFBvc2VpZG9uIEhhc2hpbmcgb2YgV2VpZ2h0cwogICAgQy0+PkM6IE1lcmtsZSBSb290IFZlcmlmaWNhdGlvbgogICAgQy0+PlA6IEdlbmVyYXRlIFByb29mIChwaV9hLCBwaV9iLCBwaV9jKQogICAgUC0+PlY6IFByb29mICsgUHVibGljIFNpZ25hbHMKICAgIFYtPj5WOiBHcm90aDE2IFZlcmlmaWNhdGlvbg==)
### Key Circuit Templates
| Template | Purpose | Source |
| :--- | :--- | :--- |
| `RSFLayerComputation` | Implements the coupling layer: $y = x \cdot \exp(s) + t$. | |
| `PoseidonChain` | Efficiently hashes large input vectors into a single field element. | |
| `VerifyMerkleProof` | Validates that weights belong to the model's committed weight tree. | |
| `RangeProof` | Ensures the injected DP noise stays within the allowed privacy bounds. | |

---

## Implementation Details

### Fixed-Point RSF Computation
Since ZK circuits operate over finite fields, the engine converts `f32` values to integers using a `FIXED_POINT_SCALE` (1,000,000). The exponential function in the RSF coupling layer is approximated using a 3rd-order Taylor expansion within the circuit:
*   **Linear Coeff**: 1
*   **Quadratic Coeff**: 0.5
*   **Cubic Coeff**: 0.166667

### Proof Aggregation
The `ProofAggregator` uses a Merkle tree to combine multiple inference proofs into a single root. This allows the system to verify a batch of $N$ inferences with $O(\log N)$ complexity.
*   **Merkle Tree**: Implemented via `ProofAggregator`.
*   **Verification**: The `verifyBatch` function checks the integrity of the entire tree.

### Commitment and Fingerprinting
The engine ensures data isolation using `DatasetFingerprint` and `CommitmentScheme`.
*   **Blake3 Commitments**: Used for fast local commitments to input data.
*   **Dataset Fingerprint**: Generates a unique 32-byte identifier for the training set to ensure the model hasn't been fine-tuned on unauthorized data.

* `src/core_relational/verified_inference_engine.zig`
* `src/core_relational/zk_verification.zig`
* `src/zk/inference_trace.circom`
* `src/core_relational/dataset_obfuscation.zig`

---

*[Back to Table of Contents](#table-of-contents) | Page 30 of 34 | Next: Security, Safety, and Formal Verification*

<a id="page-31"></a>

# Security, Safety, and Formal Verification




The JAIDE v40 codebase implements a multi-layered defense-in-depth strategy to ensure the correctness of neural-relational operations and the security of processed data. This subsystem spans from low-level runtime memory guards to high-level mathematical proofs of security models like Bell-LaPadula and Biba. By integrating formal verification directly into the `core_relational` pipeline, JAIDE ensures that information flow remains consistent with defined security policies even during complex quantum-relational reasoning.

### System Overview

The security architecture is divided into three primary domains:
1.  **Correctness & Verification**: Utilizing Hoare logic and formal invariants to prove the integrity of the `SelfSimilarRelationalGraph`.
2.  **Information Flow Control**: Enforcing lattice-based security levels and integrity levels across all system principals and objects.
3.  **Data Privacy**: Employing homomorphic encryption and dataset obfuscation to protect sensitive information during training and inference.

### Security and Verification Topology

The following diagram illustrates the relationship between the verification engine, the security policy enforcer, and the core data structures.

**Security Logic to Code Entity Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJWZXJpZmljYXRpb24gRW5naW5lIgogICAgICAgIFtGb3JtYWxWZXJpZmllcl0gLS0+IFtJbnZhcmlhbnRUeXBlXQogICAgICAgIFtGb3JtYWxWZXJpZmllcl0gLS0+IFtQcm9vZlJ1bGVdCiAgICAgICAgW0Zvcm1hbFZlcmlmaWVyXSAtLT4gW1Byb29mTm9kZV0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJTZWN1cml0eSBQb2xpY3kiCiAgICAgICAgW1NlY3VyaXR5TGV2ZWxdIC0tPiBbQmVsbExhUGFkdWxhXQogICAgICAgIFtJbnRlZ3JpdHlMZXZlbF0gLS0+IFtCaWJhTW9kZWxdCiAgICAgICAgW0FjY2Vzc1JpZ2h0XSAtLT4gW1BvbGljeUVuZm9yY2VyXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvcmUgRW50aXRpZXMiCiAgICAgICAgW1NlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoXQogICAgICAgIFtOb2RlXQogICAgICAgIFtFZGdlXQogICAgZW5kCgogICAgW0Zvcm1hbFZlcmlmaWVyXSAtLSAidmFsaWRhdGVzIiAtLT4gW1NlbGZTaW1pbGFyUmVsYXRpb25hbEdyYXBoXQogICAgW1BvbGljeUVuZm9yY2VyXSAtLSAiZ3VhcmRzIGFjY2VzcyB0byIgLS0+IFtOb2RlXQogICAgW1BvbGljeUVuZm9yY2VyXSAtLSAiY2hlY2tzIGZsb3cgb24iIC0tPiBbRWRnZV0KCiAgICBzdHlsZSBbRm9ybWFsVmVyaWZpZXJdIHN0cm9rZS13aWR0aDoycHgKICAgIHN0eWxlIFtQb2xpY3lFbmZvcmNlcl0gc3Ryb2tlLXdpZHRoOjJweA==)
---

## Formal Verification and Security Proofs

The verification subsystem provides the mathematical foundation for JAIDE's reliability. It uses a structured proof system to maintain invariants such as `CONNECTIVITY`, `SYMMETRY`, and `MEMORY_SAFETY`.

### Key Components:
*   **Security Models**: Implementation of the **Bell-LaPadula** model (no read-up, no write-down) and the **Biba** integrity model (no read-down, no write-up).
*   **Lattice-based Access Control**: Security levels ranging from `PUBLIC` to `TOP_SECRET` and integrity levels from `UNTRUSTED` to `KERNEL`.
*   **Formal Proofs**: A `FormalVerifier` that applies `ProofRule` sets (e.g., `MODUS_PONENS`, `INDUCTION`, `FRAME_RULE`) to verify graph state transitions.

For details, see [Formal Verification and Security Proofs](#9.1).

---

## Safety, Obfuscation, and C API

The safety layer provides runtime protection against common software vulnerabilities and ensures data privacy through cryptographic obfuscation.

### Key Components:
*   **Runtime Safety**: The `safety.zig` module provides checked casting (`safeIntCast`, `safePtrCast`) and secure memory operations like `secureZeroBytes` to prevent leaks of sensitive keys,.
*   **Dataset Obfuscation**: Implements **Paillier Homomorphic Encryption**, allowing mathematical operations (addition, scalar multiplication) to be performed directly on encrypted ciphertexts without decryption.
*   **Foreign Function Interface**: A C-compatible API surface defined in `c_api.zig` that allows external applications to safely interact with the JAIDE core while maintaining the security boundaries established in Zig.

**Data Protection and Safety Flow**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIFtSYXdEYXRhXSAtLT4gW0hvbW9tb3JwaGljRW5jcnlwdGlvbl0KICAgIFtIb21vbW9ycGhpY0VuY3J5cHRpb25dIC0tICJlbmNyeXB0IiAtLT4gW0NpcGhlcnRleHRdCiAgICBbQ2lwaGVydGV4dF0gLS0gImFkZC9tdWx0aXBseSIgLS0+IFtFbmNyeXB0ZWRSZXN1bHRdCiAgICBbRW5jcnlwdGVkUmVzdWx0XSAtLSAiZGVjcnlwdCIgLS0+IFtTZWN1cmVPdXRwdXRdCgogICAgc3ViZ3JhcGggIlJ1bnRpbWUgR3VhcmRzIgogICAgICAgIFtzYWZlSW50Q2FzdF0KICAgICAgICBbc2FmZVB0ckNhc3RdCiAgICAgICAgW3NlY3VyZVplcm9CeXRlc10KICAgIGVuZAoKICAgIFtDX0FQSV0gLS0gImNhbGxzIiAtLT4gW1J1bnRpbWVHdWFyZHNd)
For details, see [Safety, Obfuscation, and C API](#9.2).

---

### Security Configuration Constants

The system uses a centralized set of security parameters to define the strength of the obfuscation and the granularity of access rights.

| Constant | Value / Bit | Description |
| :--- | :--- | :--- |
| `SECURITY_PARAMETER` | 256 | Bit-length for cryptographic security |
| `ACCESS_RIGHT_READ_BIT` | 1 | Bitmask for read access |
| `ACCESS_RIGHT_ADMIN_BIT` | 16 | Bitmask for administrative privileges |
| `PRIME_P` | u256 | Pre-defined large prime for Paillier generation |


---

*[Back to Table of Contents](#table-of-contents) | Page 31 of 34 | Next: Formal Verification and Security Proofs*

<a id="page-32"></a>

# Formal Verification and Security Proofs




The JAIDE v40 system implements a rigorous multi-layered security and verification architecture. This system ensures that information flow adheres to formal security models, graph invariants are maintained during quantum-relational operations, and execution traces are cryptographically auditable. The implementation is split across the `formal_verification` module for logical correctness, `security_proofs` for access control and information flow, and `z_runtime` for safe execution.

## Security Models and Access Control

The `security_proofs.zig` module implements the **Bell-LaPadula** (Confidentiality) and **Biba** (Integrity) security models. These models are enforced through lattice-based comparisons of security and integrity levels.

### Security and Integrity Levels
The system defines discrete levels for both confidentiality and integrity:
*   **SecurityLevel**: `PUBLIC` (0) to `TOP_SECRET` (4).
*   **IntegrityLevel**: `UNTRUSTED` (0) to `KERNEL` (3).

### Formal Security Rules
The system enforces the following rules to prevent illegal information flow:
1.  **Simple Security Property (No Read Up)**: A subject at a given `SecurityLevel` cannot read an object with a higher `SecurityLevel`.
2.  **Star Property (No Write Down)**: A subject cannot write to an object with a lower `SecurityLevel`.
3.  **Biba Integrity (No Read Down/No Write Up)**: Subjects cannot read from lower integrity levels or write to higher integrity levels.

### Security Model Data Flow
The following diagram illustrates how `SecurityLevel` and `IntegrityLevel` interact within the `SecurityProofsConfig` to validate access.

**Security Validation Logic**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJTdWJqZWN0X0NvbnRleHQiCiAgICAgICAgU19MVkxbIlNlY3VyaXR5TGV2ZWwiXQogICAgICAgIElfTFZMWyJJbnRlZ3JpdHlMZXZlbCJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiT2JqZWN0X0NvbnRleHQiCiAgICAgICAgT19MVkxbIlNlY3VyaXR5TGV2ZWwiXQogICAgICAgIE9JX0xWTFsiSW50ZWdyaXR5TGV2ZWwiXQogICAgZW5kCgogICAgU19MVkwgLS0gImRvbWluYXRlcygpIiAtLT4gT19MVkwKICAgIE9fTFZMIC0tICJpc0RvbWluYXRlZEJ5KCkiIC0tPiBTX0xWTAogICAgSV9MVkwgLS0gImdyZWF0ZXJUaGFuT3JFcXVhbCgpIiAtLT4gT0lfTFZMCgogICAgU19MVkwgLS0+IEJMUF9DaGVja1siQmVsbC1MYVBhZHVsYSBDaGVjayJdCiAgICBPX0xWTCAtLT4gQkxQX0NoZWNrCiAgICBJX0xWTCAtLT4gQmliYV9DaGVja1siQmliYSBDaGVjayJdCiAgICBPSV9MVkwgLS0+IEJpYmFfQ2hlY2sKCiAgICBCTFBfQ2hlY2sgLS0gIlZpb2xhdGlvbiIgLS0+IFNFWyJTZWN1cml0eUVycm9yLkJlbGxMYVBhZHVsYVZpb2xhdGlvbiJdCiAgICBCaWJhX0NoZWNrIC0tICJWaW9sYXRpb24iIC0tPiBJRVsiU2VjdXJpdHlFcnJvci5CaWJhVmlvbGF0aW9uIl0KCiAgICBzdHlsZSBTRSBzdHJva2UtZGFzaGFycmF5OiA1IDUKICAgIHN0eWxlIElFIHN0cm9rZS1kYXNoYXJyYXk6IDUgNQ==)
## Formal Verification and Invariants

The `formal_verification.zig` module provides a framework for proving the correctness of the `SelfSimilarRelationalGraph`. It uses a set of `InvariantType` definitions to monitor the health and mathematical consistency of the NSIR graph.

### Invariant Types and Priorities
The system prioritizes safety-critical invariants over structural ones:
*   **MEMORY_SAFETY (Priority 10)**: Ensures no buffer overflows or invalid pointer dereferences.
*   **TYPE_SAFETY (Priority 9)**: Validates quantum and relational type conversions.
*   **CONNECTIVITY & COHERENCE**: Ensures graph topology and quantum phase consistency.

### Proof Rules
The module implements a logic engine based on `ProofRule`. These rules allow the system to derive safety properties from axioms:
*   **MODUS_PONENS**: Requires 2 premises.
*   **TEMPORAL_INDUCTION**: Used for verifying state transitions over time.
*   **LOOP_INVARIANT**: Ensures graph traversal stability.

## Z-Runtime Safety Layer

The `z_runtime.zig` module acts as a managed execution environment for relational operations. It maintains a complete `ExecutionHistory` to provide an auditable trail of every transformation within the system.

### Execution Auditing
Every action taken by the `ZRuntime` is recorded as an `ExecutionHistoryEntry`. This includes:
*   **Action Types**: `create_variable`, `relational_operation`, `entangle_variables`, `measure`.
*   **Metadata**: Timestamps (nanosecond precision), primary/secondary targets, and result values.

### Variable Lifecycle and Ownership
The `ZVariable` struct encapsulates a `SelfSimilarRelationalGraph` and its associated `RelationalQuantumLogic`. It tracks its own `history` and `creation_order` to ensure that state changes can be verified against the formal rules defined in the security modules.

### Runtime Entity Mapping
This diagram maps the natural language concepts of "Variable Execution" to the internal Zig entities.

**Z-Runtime Execution Flow**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJOYXR1cmFsX0xhbmd1YWdlX1NwYWNlIgogICAgICAgIFZBUlsiVmFyaWFibGUiXQogICAgICAgIEFDVFsiT3BlcmF0aW9uIl0KICAgICAgICBBVURJVFsiQXVkaXQgVHJhaWwiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGVfRW50aXR5X1NwYWNlIgogICAgICAgIFpWQVJbIlpWYXJpYWJsZSJdCiAgICAgICAgRVhfQUNUWyJFeGVjdXRpb25BY3Rpb24iXQogICAgICAgIEhJU1RbIkV4ZWN1dGlvbkhpc3RvcnlFbnRyeSJdCiAgICAgICAgTE9HSUNbIlJlbGF0aW9uYWxRdWFudHVtTG9naWMiXQogICAgZW5kCgogICAgVkFSIC0uLT4gWlZBUgogICAgQUNUIC0uLT4gRVhfQUNUCiAgICBBVURJVCAtLi0+IEhJU1QKCiAgICBaVkFSIC0tICJleGVjdXRlcyIgLS0+IEVYX0FDVAogICAgRVhfQUNUIC0tICJtb2RpZmllcyIgLS0+IExPR0lDCiAgICBFWF9BQ1QgLS0gImxvZ3MgdG8iIC0tPiBISVNUCiAgICBaVkFSIC0tICJjb250YWlucyIgLS0+IEhJU1Q=)
## Cryptographic Integration

The verification layer utilizes high-performance cryptographic primitives to ensure the integrity of proofs and security labels.

| Component | Primitive | Purpose |
| :--- | :--- | :--- |
| **CommitmentScheme** | `Blake3` | Fast, secure state commitments for ZK proofs. |
| **Security Proofs** | `Sha256` / `Sha512` | Hashing of security descriptors and integrity labels. |
| **Verification** | `timingSafeEql` | Constant-time comparison to prevent side-channel attacks. |

*   **Hashing Implementation**: Uses `std.crypto.hash.sha2.Sha256` and `Sha512` for generating unique identifiers for security contexts.
*   **Homomorphic Integration**: While standard tensors are processed normally, the verification layer supports integration with Paillier homomorphic encryption for privacy-preserving verification of model weights (referenced in `SecurityError.CryptographicError`).


---

*[Back to Table of Contents](#table-of-contents) | Page 32 of 34 | Next: Safety, Obfuscation, and C API*

<a id="page-33"></a>

# Safety, Obfuscation, and C API




This section covers the subsystems responsible for runtime integrity, data privacy, and interoperability. The `safety` module provides robust runtime guards to prevent common memory and arithmetic errors. The `dataset_obfuscation` module implements privacy-preserving techniques, including Paillier homomorphic encryption, for handling sensitive training data. Finally, the `c_api.zig` provides a Foreign Function Interface (FFI) for integrating JAIDE’s NSIR and optimization capabilities into external C/C++ environments.

## Safety and Runtime Guards

The safety module implements a comprehensive suite of utilities designed to enforce system integrity at runtime. These utilities are used across the codebase to mitigate risks associated with manual memory management and low-level bit manipulation.

### Core Safety Functions
*   **Safe Casting**: `safeIntCast` and `safeUsizeToInt` perform bounds checking to prevent `IntegerOverflow` and `IntegerUnderflow`.
*   **Pointer Validation**: `safePtrCast` ensures that pointers are non-null, correctly aligned for the target type, and possess valid provenance.
*   **Memory Zeroing**: `secureZeroBytes` and `secureZeroSlice` use volatile writes to ensure sensitive data is cleared from memory and not optimized away by the compiler.
*   **Constant-Time Comparison**: `secureCompare` provides timing-attack resistant byte comparison.

### Secure Utilities
The module also provides a `SecureRng` wrapper around `std.crypto.random` and a `MonotonicClock` for high-precision, drift-resistant timing measurements.

---

## Dataset Obfuscation and Homomorphic Encryption

JAIDE utilizes the `dataset_obfuscation` module to handle sensitive datasets without exposing raw values during certain processing stages. This is achieved primarily through a custom implementation of the Paillier cryptosystem.

### Paillier Cryptosystem Implementation
The `PaillierKeyPair` stores the public and private components ($n, g, \lambda, \mu$) required for asymmetric homomorphic encryption.

*   **Encryption**: `encrypt` converts an `i64` plaintext into a `u512` ciphertext. It uses a 256-bit sign-bit encoding to support negative integers.
*   **Decryption**: `decrypt` recovers the original integer using the private $\lambda$ and $\mu$ parameters.
*   **Homomorphic Operations**:
    *   `add`: Multiplies two ciphertexts to produce a ciphertext of their sum.
    *   `multiplyScalar`: Raises a ciphertext to a scalar power to produce a ciphertext of the product.

### Obfuscation Data Flow
The following diagram illustrates how data is transformed from plaintext to an obfuscated state for processing.

**Data Obfuscation Pipeline**
![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIC8gUmF3IFNwYWNlIgogICAgICAgICJSYXdEYXRhc2V0IlsiUmF3IERhdGFzZXQgKGk2NCkiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGUgRW50aXR5IFNwYWNlOiBkYXRhc2V0X29iZnVzY2F0aW9uLnppZyIKICAgICAgICAiUmF3RGF0YXNldCIgLS0+ICJIRV9Jbml0IlsiSG9tb21vcnBoaWNFbmNyeXB0aW9uLmluaXQoKSJdCiAgICAgICAgIkhFX0luaXQiIC0tPiAiS2V5R2VuIlsiUGFpbGxpZXJLZXlQYWlyLmdlbmVyYXRlKCkiXQogICAgICAgICJLZXlHZW4iIC0tPiAiRW5jT3AiWyJIb21vbW9ycGhpY0VuY3J5cHRpb24uZW5jcnlwdCgpIl0KICAgICAgICAiRW5jT3AiIC0tPiAiQ2lwaGVydGV4dCJbInU1MTIgQ2lwaGVydGV4dCJdCiAgICAgICAgIkNpcGhlcnRleHQiIC0tPiAiSG9tQWRkIlsiSG9tb21vcnBoaWNFbmNyeXB0aW9uLmFkZCgpIl0KICAgICAgICAiSG9tQWRkIiAtLT4gIkRlY09wIlsiSG9tb21vcnBoaWNFbmNyeXB0aW9uLmRlY3J5cHQoKSJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiU2VjdXJpdHkgTGF5ZXIiCiAgICAgICAgIkRlY09wIiAtLT4gIlNlY1plcm8iWyJzZWN1cmVaZXJvQnl0ZXMoKSJdCiAgICBlbmQ=)
---

## C API and FFI Bridge

The `c_api.zig` file defines the external interface for JAIDE, allowing C-compatible languages to interact with the `nsir_core` and the `EntangledStochasticSymmetryOptimizer`.

### Handle-Based Architecture
The API uses opaque pointers to manage internal Zig state safely:
*   `CGraph`: An opaque handle to a `GraphContext`, which wraps the `SelfSimilarRelationalGraph` and a mutex for thread safety.
*   `COptimizer`: An opaque handle to the `EntangledStochasticSymmetryOptimizer`.

### Optimization and Statistics
The `EntangledStochasticSymmetryOptimizer` implements a simulated annealing approach to graph energy minimization. It tracks its progress via `OptimizationStatistics`, which records iterations, acceptance rates, and energy levels.

### C API Integration Mapping
The following diagram bridges the C-level function calls to the internal Zig implementations.

**C API to Internal Logic Mapping**
![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJFeHRlcm5hbCBDIC8gQysrIFNwYWNlIgogICAgICAgICJDX0NhbGwiWyJqYWlkZV9ncmFwaF9hZGRfbm9kZSgpIl0KICAgICAgICAiQ19PcHQiWyJqYWlkZV9vcHRpbWl6ZXJfcnVuKCkiXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkNvZGUgRW50aXR5IFNwYWNlOiBjX2FwaS56aWciCiAgICAgICAgIkNfQ2FsbCIgLS0+ICJDR3JhcGhfVG8iWyJDR3JhcGgudG9JbnRlcm5hbCgpIl0KICAgICAgICAiQ0dyYXBoX1RvIiAtLT4gIkN0eCJbIkdyYXBoQ29udGV4dCJdCiAgICAgICAgIkN0eCIgLS0+ICJOU0lSX0FkZCJbIm5zaXJfY29yZS5hZGROb2RlKCkiXQogICAgICAgIAogICAgICAgICJDX09wdCIgLS0+ICJDT3B0X1RvIlsiQ09wdGltaXplci50b0ludGVybmFsKCkiXQogICAgICAgICJDT3B0X1RvIiAtLT4gIkVTU08iWyJFbnRhbmdsZWRTdG9jaGFzdGljU3ltbWV0cnlPcHRpbWl6ZXIiXQogICAgICAgICJFU1NPIiAtLT4gIlN0YXRzIlsiT3B0aW1pemF0aW9uU3RhdGlzdGljcy5yZWNvcmRBY2NlcHRlZCgpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJFcnJvciBIYW5kbGluZyIKICAgICAgICAiTlNJUl9BZGQiIC0tPiAiRXJyQ29kZSJbIkpBSURFX0VSUk9SX0FMTE9DQVRJT04gKC0yKSJdCiAgICBlbmQ=)

### Error Codes
The API returns `c_int` status codes to indicate success or specific failure modes:
| Constant | Value | Description |
| :--- | :--- | :--- |
| `JAIDE_SUCCESS` | 0 | Operation completed successfully |
| `JAIDE_ERROR_NULL_POINTER` | -1 | A null pointer was passed to the API |
| `JAIDE_ERROR_ALLOCATION` | -2 | Memory allocation failed |
| `JAIDE_ERROR_NODE_NOT_FOUND` | -3 | Target node does not exist in the graph |
| `JAIDE_ERROR_OPTIMIZATION_FAILED` | -6 | The optimizer failed to converge |


---

*[Back to Table of Contents](#table-of-contents) | Page 33 of 34 | Next: Glossary*

<a id="page-34"></a>

# Glossary




This glossary defines the technical terminology, domain-specific concepts, and architectural primitives used in the JAIDE v40 codebase. JAIDE is a Large Language Model based on the **Reversible Scatter Flow (RSF)** paradigm, designed for high-performance execution across CPUs, GPUs, and quantum relational graphs.

---

## Core Architectural Terms

### RSF (Reversible Scatter Flow)
The fundamental neural architecture of JAIDE. Unlike Transformers or CNNs, RSF is built on **bijective coupling layers** that ensure every forward operation has an exact algebraic inverse. This allows for $O(1)$ memory complexity during backpropagation because activations can be reconstructed on-the-fly from outputs.
*   **Implementation:** `LayerCore` struct in.
*   **Key Functions:** `forwardInPlace` and `inverseInPlace`.

### NSIR (Non-linear Self-Similar Information Retrieval)
A quantum-relational graph system used for hierarchical reasoning. It represents knowledge as a graph of nodes (with quantum states) and edges (with entanglement and fractal properties).
*   **Implementation:** `SelfSimilarRelationalGraph` in.
*   **Concepts:** Qubits, entanglement, and complex amplitudes are used to represent relational confidence.

### OFTB (Orthogonal Fractal Transform Block)
A parameter-less mixer block that uses Haar-wavelet butterfly transforms to distribute information across the tensor dimensions. It replaces the self-attention mechanism found in Transformers.
*   **Implementation:** `rsf_scatter` kernel in.
*   **Logic:** Uses an `inv_sqrt2` (1/√2) scaling factor to maintain variance.

### SFD (Stochastic Fisher Diagonal)
A second-order optimizer used for training RSF models. It estimates the Fisher Information Matrix diagonal to perform natural gradient updates, combined with mixed-precision support (FP4 to FP64).
*   **Implementation:** `SFDOptimizer` in.
*   **Components:** Includes `MixedPrecisionTrainer` and `B200MemoryManager`.

---

## Data Structures and Primitives

| Term | Definition | File Reference |
| :--- | :--- | :--- |
| **Tensor** | The primary data structure for multi-dimensional arrays, supporting copy-on-write (COW) and reference counting. | |
| **MGT** | Morpheme-Guided Tokenizer. A three-tier pipeline: special tokens → morphological decomposition → BPE fallback. | |
| **SSI** | Structured Sequence Index. A 64-bucket hash tree used for Hamming-distance similarity search of sequence segments. | |
| **LayerCore** | The "unit of computation" in JAIDE, containing 4 trainable tensors: `s_weight`, `t_weight`, `s_bias`, `t_bias`. | |
| **Qubit** | A representation of a node's state in NSIR, defined by two complex amplitudes (alpha and beta). | |

---

## Processing Logic Diagrams

### Natural Language to Code Entity Mapping: Inference Flow
The following diagram maps high-level inference concepts to the specific classes and functions in the codebase.

![Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIHN1YmdyYXBoICJOYXR1cmFsIExhbmd1YWdlIFNwYWNlIgogICAgICAgIEFbIlVzZXIgSW5wdXQgU3RyaW5nIl0KICAgICAgICBCWyJUb2tlbiBTdHJlYW0iXQogICAgICAgIENbIk5ldXJhbCBUcmFuc2Zvcm1hdGlvbiJdCiAgICAgICAgRFsiS25vd2xlZGdlIFJldHJpZXZhbCJdCiAgICBlbmQKCiAgICBzdWJncmFwaCAiQ29kZSBFbnRpdHkgU3BhY2UiCiAgICAgICAgQSAtLT58InRva2VuaXplKCkifCBFWyJNR1QgKG1ndC56aWcpIl0KICAgICAgICBFIC0tPiBGWyJUZW5zb3IgKHRlbnNvci56aWcpIl0KICAgICAgICBGIC0tPiBHWyJMYXllckNvcmUuZm9yd2FyZEluUGxhY2UgKHJzZi56aWcpIl0KICAgICAgICBHIC0tPiBIWyJyc2ZfZmxvdyAoZnV0aGFya19rZXJuZWxzLmZ1dCkiXQogICAgICAgIEggLS0+IElbIlNTSS5yZXRyaWV2ZVRvcEsgKHNzaS56aWcpIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJIYXJkd2FyZSBFeGVjdXRpb24iCiAgICAgICAgSCAtLT4gSlsiR1BVIChjdWRhX2JpbmRpbmdzLnppZykiXQogICAgZW5k)
---

### Knowledge Graph Evolution: CREV Pipeline
The CREV (Collective Relational Evolution and Validation) pipeline bridges text ingestion with the NSIR quantum graph.

![Diagram](https://mermaid.ink/img/Z3JhcGggTFIKICAgIHN1YmdyYXBoICJJbmdlc3Rpb24iCiAgICAgICAgVFsiUmF3IFRleHQiXSAtLT4gUFsiQ1JFVlBpcGVsaW5lLnByb2Nlc3NTdHJlYW0iXQogICAgZW5kCgogICAgc3ViZ3JhcGggIkV4dHJhY3Rpb24gJiBWYWxpZGF0aW9uIgogICAgICAgIFAgLS0+IFJUWyJSZWxhdGlvbmFsVHJpcGxldCAoY3Jldl9waXBlbGluZS56aWcpIl0KICAgICAgICBSVCAtLT4gVlsiVmFsaWRhdGlvblN0YWdlIl0KICAgIGVuZAoKICAgIHN1YmdyYXBoICJHcmFwaCBJbnRlZ3JhdGlvbiIKICAgICAgICBWIC0tPiBOWyJTZWxmU2ltaWxhclJlbGF0aW9uYWxHcmFwaC5hZGROb2RlIChuc2lyX2NvcmUuemlnKSJdCiAgICAgICAgTiAtLT4gUVsiUXViaXQubm9ybWFsaXplSW5QbGFjZSJdCiAgICAgICAgUSAtLT4gRVsiUmVhc29uaW5nT3JjaGVzdHJhdG9yLnBlcnR1cmJMb2NhbE5vZGVzIl0KICAgIGVuZA==)
---

## Memory and Hardware Concepts

### Arena / Slab / Buddy Allocators
JAIDE uses a hierarchy of custom allocators to minimize fragmentation and maximize throughput during high-concurrency inference.
*   **Arena:** Fast, linear allocation for per-request lifecycles.
*   **Buddy:** Used for dynamic, power-of-two sized blocks in the NSIR graph.

### Futhark Kernels
The GPU-accelerated backend. Futhark code is compiled to OpenCL or CUDA and handles the heavy lifting of `rsf_flow` and `natural_gradient` updates.
*   **Butterfly Mixing:** Implementation of the Haar-like scatter operation.
*   **Fisher Update:** Moving average of squared gradients for the SFD optimizer.

### GPUCoordinator
Manages multi-GPU communication using NCCL (NVIDIA Collective Communications Library). It handles synchronization barriers and collective operations like `allReduce` for weight-delta averaging.
*   **Implementation:** `GPUCoordinator` in.
*   **Key Function:** `ncclCommInitRank` for initializing distributed groups.

---

## Mathematical Symbols in Code

*   **$s\_weight$ ($W_s$):** Scale weight matrix used to compute the affine scaling factor $exp(W_s \cdot x_2 + b_s)$.
*   **$t\_weight$ ($W_t$):** Translation weight matrix used for the affine shift $W_t \cdot y_1 + b_t$.
*   **$inv\_sqrt2$ (1/√2):** The fractal scaling constant used in OFTB to preserve energy during butterfly transforms.
*   **Complex Amplitudes ($a, b$):** Represent the probability of a node being in state $|0\rangle$ or $|1\rangle$ in the NSIR graph.

---

---

*[Back to Table of Contents](#table-of-contents) | Page 34 of 34*

