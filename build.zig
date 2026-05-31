const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu_enabled = b.option(bool, "gpu", "Enable GPU/CUDA acceleration via Futhark CUDA backend") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "gpu_acceleration", gpu_enabled);

    const futhark_c = b.path("src/hw/accel/futhark_kernels.c");
    const futhark_gpu_c = b.path("src/hw/accel/main_gpu.c");
    const futhark_include = b.path("src/hw/accel");

    const inference_server_exe = b.addExecutable(.{
        .name = "jaide-inference-server",
        .root_source_file = b.path("src/inference_server_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_server_exe.linkLibC();
    inference_server_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    inference_server_exe.addIncludePath(futhark_include);
    inference_server_exe.root_module.addOptions("build_options", build_options);
    b.installArtifact(inference_server_exe);

    if (gpu_enabled) {
        const distributed_futhark_exe = b.addExecutable(.{
            .name = "jaide-distributed-futhark",
            .root_source_file = b.path("src/main_distributed_futhark.zig"),
            .target = target,
            .optimize = optimize,
        });
        distributed_futhark_exe.linkLibC();
        distributed_futhark_exe.addCSourceFile(.{ .file = futhark_gpu_c, .flags = &.{"-O2"} });
        distributed_futhark_exe.addIncludePath(futhark_include);
        distributed_futhark_exe.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
        distributed_futhark_exe.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        distributed_futhark_exe.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64/stubs" });
        distributed_futhark_exe.linkSystemLibrary("cuda");
        distributed_futhark_exe.linkSystemLibrary("cudart");
        distributed_futhark_exe.linkSystemLibrary("nvrtc");
        distributed_futhark_exe.linkSystemLibrary("nccl");
        distributed_futhark_exe.root_module.addOptions("build_options", build_options);
        b.installArtifact(distributed_futhark_exe);

        const distributed_futhark_install = b.addInstallArtifact(distributed_futhark_exe, .{});
        const distributed_futhark_step = b.step("distributed-futhark", "Build only the Futhark-accelerated distributed trainer");
        distributed_futhark_step.dependOn(&distributed_futhark_install.step);
    }

    const tensor_tests = b.addTest(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);
    const tensor_test_step = b.step("test-tensor", "Run tensor tests");
    tensor_test_step.dependOn(&run_tensor_tests.step);

    const memory_tests = b.addTest(.{
        .root_source_file = b.path("src/core/memory.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_memory_tests = b.addRunArtifact(memory_tests);
    const memory_test_step = b.step("test-memory", "Run memory tests");
    memory_test_step.dependOn(&run_memory_tests.step);

    // Benchmark dependency module (rooted at src/ for Zig module path resolution)
    const bench_deps = b.createModule(.{
        .root_source_file = b.path("src/_bench_deps.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_deps.addOptions("build_options", build_options);

    // Benchmark targets
    const bench_step = b.step("bench", "Run all benchmarks");

    const bench_sources = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "bench-rsf", .path = "src/tests/bench_rsf.zig" },
        .{ .name = "bench-matmul", .path = "src/tests/bench_matmul.zig" },
        .{ .name = "bench-tensor-ops", .path = "src/tests/bench_tensor_ops.zig" },
        .{ .name = "bench-sfd", .path = "src/tests/bench_sfd.zig" },
    };

    inline for (bench_sources) |src| {
        const exe = b.addExecutable(.{
            .name = src.name,
            .root_source_file = b.path(src.path),
            .target = target,
            .optimize = optimize,
        });
        exe.linkLibC();
        exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        exe.addIncludePath(futhark_include);
        exe.root_module.addOptions("build_options", build_options);
        exe.root_module.addImport("deps", bench_deps);
        b.installArtifact(exe);
        const run = b.addRunArtifact(exe);
        bench_step.dependOn(&run.step);
    }
}
