const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;

const nsir = @import("nsir_core.zig");
const esso = @import("esso_optimizer.zig");
const chaos = @import("chaos_core.zig");
const fnds = @import("fnds.zig");
const quantum = @import("quantum_logic.zig");
const Complex = std.math.Complex;

const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const Qubit = nsir.Qubit;
const EntangledStochasticSymmetryOptimizer = esso.EntangledStochasticSymmetryOptimizer;
const ChaosCoreKernel = chaos.ChaosCoreKernel;
const FractalTree = fnds.FractalTree;
const SymmetryPattern = esso.SymmetryPattern;

pub const PatternId = [32]u8;

pub const ThoughtLevel = enum(u8) {
    local = 0,
    global = 1,
    meta = 2,

    pub fn toString(self: ThoughtLevel) []const u8 {
        return switch (self) {
            .local => "local",
            .global => "global",
            .meta => "meta",
        };
    }
};

pub const ReasoningPhase = struct {
    phase_id: u64,
    level: ThoughtLevel,
    inner_iterations: usize,
    outer_iterations: usize,
    target_energy: f64,
    current_energy: f64,
    previous_energy: f64,
    convergence_threshold: f64,
    phase_start_time: i64,
    phase_end_time: i64,
    pattern_captures: ArrayList(PatternId),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: ThoughtLevel, inner: usize, outer: usize, phase_id: u64) Self {
        return Self{
            .phase_id = phase_id,
            .level = level,
            .inner_iterations = inner,
            .outer_iterations = outer,
            .target_energy = 0.1,
            .current_energy = 1e6,
            .previous_energy = 1e6,
            .convergence_threshold = 1e-6,
            .phase_start_time = @as(i64, @intCast(std.time.nanoTimestamp())),
            .phase_end_time = 0,
            .pattern_captures = ArrayList(PatternId).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pattern_captures.deinit();
    }

    pub fn recordPattern(self: *Self, pattern_id: PatternId) !void {
        try self.pattern_captures.append(pattern_id);
    }

    pub fn hasConverged(self: *const Self) bool {
        const delta = @abs(self.current_energy - self.previous_energy);
        const denom = @max(@abs(self.previous_energy), 1.0);
        return (delta / denom) < self.convergence_threshold;
    }

    pub fn updateEnergy(self: *Self, new_energy: f64) void {
        self.previous_energy = self.current_energy;
        self.current_energy = new_energy;
    }

    pub fn finalize(self: *Self) void {
        self.phase_end_time = @as(i64, @intCast(std.time.nanoTimestamp()));
    }

    pub fn getDuration(self: *const Self) i64 {
        if (self.phase_end_time > 0) {
            return self.phase_end_time - self.phase_start_time;
        }
        return @as(i64, @intCast(std.time.nanoTimestamp())) - self.phase_start_time;
    }
};

pub const OrchestratorStatistics = struct {
    total_phases: usize,
    local_phases: usize,
    global_phases: usize,
    meta_phases: usize,
    total_inner_loops: usize,
    total_outer_loops: usize,
    average_convergence_time: f64,
    best_energy_achieved: f64,
    patterns_discovered: usize,
    orchestration_start_time: i64,

    pub fn init() OrchestratorStatistics {
        return OrchestratorStatistics{
            .total_phases = 0,
            .local_phases = 0,
            .global_phases = 0,
            .meta_phases = 0,
            .total_inner_loops = 0,
            .total_outer_loops = 0,
            .average_convergence_time = 0.0,
            .best_energy_achieved = std.math.inf(f64),
            .patterns_discovered = 0,
            .orchestration_start_time = @as(i64, @intCast(std.time.nanoTimestamp())),
        };
    }

    pub fn recordPhase(self: *OrchestratorStatistics, phase: *const ReasoningPhase) void {
        self.total_phases += 1;
        switch (phase.level) {
            .local => self.local_phases += 1,
            .global => self.global_phases += 1,
            .meta => self.meta_phases += 1,
        }
        self.total_inner_loops += phase.inner_iterations;
        self.total_outer_loops += phase.outer_iterations;
        if (phase.current_energy < self.best_energy_achieved) {
            self.best_energy_achieved = phase.current_energy;
        }
        self.patterns_discovered += phase.pattern_captures.items.len;

        const duration = @as(f64, @floatFromInt(phase.getDuration()));
        const n = @as(f64, @floatFromInt(self.total_phases));
        self.average_convergence_time += (duration - self.average_convergence_time) / n;
    }
};

pub const ReasoningResult = struct {
    best_energy: f64,
    modulation_factor: f64,
    phases_completed: usize,
    patterns_found: usize,
};

pub const ReasoningOrchestrator = struct {
    graph: *SelfSimilarRelationalGraph,
    esso: *EntangledStochasticSymmetryOptimizer,
    chaos_kernel: *ChaosCoreKernel,
    phase_history: ArrayList(ReasoningPhase),
    statistics: OrchestratorStatistics,
    fast_inner_steps: usize,
    slow_outer_steps: usize,
    hierarchical_depth: usize,
    perturb_node_limit: usize,
    update_edge_limit: usize,
    transform_node_limit: usize,
    next_phase_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        graph: *SelfSimilarRelationalGraph,
        esso_opt: *EntangledStochasticSymmetryOptimizer,
        kernel: *ChaosCoreKernel,
    ) Self {
        return Self{
            .graph = graph,
            .esso = esso_opt,
            .chaos_kernel = kernel,
            .phase_history = ArrayList(ReasoningPhase).init(allocator),
            .statistics = OrchestratorStatistics.init(),
            .fast_inner_steps = 50,
            .slow_outer_steps = 10,
            .hierarchical_depth = 3,
            .perturb_node_limit = 10,
            .update_edge_limit = 10,
            .transform_node_limit = 5,
            .next_phase_id = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.phase_history.items) |*phase| {
            phase.deinit();
        }
        self.phase_history.deinit();
    }

    pub fn setParameters(self: *Self, inner: usize, outer: usize, depth: usize) void {
        self.fast_inner_steps = inner;
        self.slow_outer_steps = outer;
        self.hierarchical_depth = depth;
    }

    pub fn setProcessingLimits(self: *Self, perturb_nodes: usize, update_edges: usize, transform_nodes: usize) void {
        self.perturb_node_limit = perturb_nodes;
        self.update_edge_limit = update_edges;
        self.transform_node_limit = transform_nodes;
    }

    fn allocatePhaseId(self: *Self) u64 {
        const id = self.next_phase_id;
        self.next_phase_id += 1;
        return id;
    }

    fn executeLocalPhaseInternal(self: *Self, record: bool) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .local, self.fast_inner_steps, 1, self.allocatePhaseId());
        errdefer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        var iteration: usize = 0;
        while (iteration < self.fast_inner_steps) : (iteration += 1) {
            try self.perturbLocalNodes();
            try self.updateLocalEdges();

            const energy = self.computeGraphEnergy();
            phase.updateEnergy(energy);

            if (iteration > 0 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;

        if (record) {
            self.statistics.recordPhase(&phase);
            try self.phase_history.append(phase);
        } else {
            phase.deinit();
        }

        return final_energy;
    }

    pub fn executeLocalPhase(self: *Self) !f64 {
        return self.executeLocalPhaseInternal(true);
    }

    fn perturbLocalNodes(self: *Self) !void {
        var node_iter = self.graph.nodes.iterator();
        var count: usize = 0;
        while (node_iter.next()) |entry| {
            if (count >= self.perturb_node_limit) break;
            var node = entry.value_ptr;
            const perturbation_a = (self.esso.prng.random().float(f64) - 0.5) * 0.1;
            const perturbation_b = (self.esso.prng.random().float(f64) - 0.5) * 0.1;
            node.phase += (perturbation_a + perturbation_b) * 0.5;
            const perturb_scale = 0.01;
            var new_a_re = node.qubit.a.re + perturbation_a * perturb_scale;
            var new_a_im = node.qubit.a.im + perturbation_a * perturb_scale;
            var new_b_re = node.qubit.b.re + perturbation_b * perturb_scale;
            var new_b_im = node.qubit.b.im + perturbation_b * perturb_scale;
            const mag = std.math.sqrt(new_a_re * new_a_re + new_a_im * new_a_im + new_b_re * new_b_re + new_b_im * new_b_im);
            if (mag > 1e-12) {
                node.qubit.a.re = new_a_re / mag;
                node.qubit.a.im = new_a_im / mag;
                node.qubit.b.re = new_b_re / mag;
                node.qubit.b.im = new_b_im / mag;
            } else {
                node.qubit.a.re = 1.0;
                node.qubit.a.im = 0.0;
                node.qubit.b.re = 0.0;
                node.qubit.b.im = 0.0;
            }
            count += 1;
        }
    }

    fn updateLocalEdges(self: *Self) !void {
        var edge_iter = self.graph.edges.iterator();
        var count: usize = 0;
        while (edge_iter.next()) |entry| {
            if (count >= self.update_edge_limit) break;
            for (entry.value_ptr.items) |*edge| {
                const delta = (self.esso.prng.random().float(f64) - 0.5) * 0.05;
                edge.weight = std.math.clamp(edge.weight + delta, 0.0, 1.0);
                const corr_delta = delta * 0.1;
                edge.quantum_correlation.re += corr_delta;
                edge.quantum_correlation.im += corr_delta * 0.5;
            }
            count += 1;
        }
    }

    fn executeGlobalPhaseInternal(self: *Self, record: bool) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .global, self.fast_inner_steps, self.slow_outer_steps, self.allocatePhaseId());
        errdefer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        var outer_iteration: usize = 0;
        while (outer_iteration < self.slow_outer_steps) : (outer_iteration += 1) {
            try self.transformSymmetryPatterns();
            try self.rebalanceFractalStructures();

            var inner: usize = 0;
            while (inner < self.fast_inner_steps) : (inner += 1) {
                try self.chaos_kernel.executeCycle();
            }

            const energy = self.computeGraphEnergy();
            phase.updateEnergy(energy);

            if (outer_iteration > 0 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;

        if (record) {
            self.statistics.recordPhase(&phase);
            try self.phase_history.append(phase);
        } else {
            phase.deinit();
        }

        return final_energy;
    }

    pub fn executeGlobalPhase(self: *Self) !f64 {
        return self.executeGlobalPhaseInternal(true);
    }

    fn transformSymmetryPatterns(self: *Self) !void {
        const transforms = try self.esso.detectSymmetries(self.graph);
        defer self.allocator.free(transforms);

        const epsilon: f64 = 1e-10;

        for (transforms) |transform| {
            var node_iter = self.graph.nodes.iterator();
            var count: usize = 0;
            while (node_iter.next()) |entry| {
                if (count >= self.transform_node_limit) break;
                const node = entry.value_ptr;
                const quantum_state_a = quantum.QuantumState{
                    .amplitude_real = node.qubit.a.re,
                    .amplitude_imag = node.qubit.a.im,
                    .phase = node.phase,
                    .entanglement_degree = 0.0,
                };
                const transformed_a = transform.applyToQuantumState(&quantum_state_a);
                const quantum_state_b = quantum.QuantumState{
                    .amplitude_real = node.qubit.b.re,
                    .amplitude_imag = node.qubit.b.im,
                    .phase = node.phase,
                    .entanglement_degree = 0.0,
                };
                const transformed_b = transform.applyToQuantumState(&quantum_state_b);
                var new_a_re = if (@abs(transformed_a.amplitude_real) < epsilon) epsilon else transformed_a.amplitude_real;
                var new_a_im = if (@abs(transformed_a.amplitude_imag) < epsilon) epsilon else transformed_a.amplitude_imag;
                var new_b_re = if (@abs(transformed_b.amplitude_real) < epsilon) epsilon else transformed_b.amplitude_real;
                var new_b_im = if (@abs(transformed_b.amplitude_imag) < epsilon) epsilon else transformed_b.amplitude_imag;
                const mag = std.math.sqrt(new_a_re * new_a_re + new_a_im * new_a_im + new_b_re * new_b_re + new_b_im * new_b_im);
                if (mag > 1e-12) {
                    node.qubit.a.re = new_a_re / mag;
                    node.qubit.a.im = new_a_im / mag;
                    node.qubit.b.re = new_b_re / mag;
                    node.qubit.b.im = new_b_im / mag;
                } else {
                    node.qubit.a.re = 1.0;
                    node.qubit.a.im = 0.0;
                    node.qubit.b.re = 0.0;
                    node.qubit.b.im = 0.0;
                }
                node.phase = transformed_a.phase;
                count += 1;
            }
        }

        var norm_iter = self.graph.nodes.iterator();
        while (norm_iter.next()) |entry| {
            var node = entry.value_ptr;
            const re_a = node.qubit.a.re;
            const im_a = node.qubit.a.im;
            const re_b = node.qubit.b.re;
            const im_b = node.qubit.b.im;
            const mag = std.math.sqrt(re_a * re_a + im_a * im_a + re_b * re_b + im_b * im_b);
            if (mag > 1e-12) {
                node.qubit.a.re = re_a / mag;
                node.qubit.a.im = im_a / mag;
                node.qubit.b.re = re_b / mag;
                node.qubit.b.im = im_b / mag;
            } else {
                node.qubit.a.re = 1.0;
                node.qubit.a.im = 0.0;
                node.qubit.b.re = 0.0;
                node.qubit.b.im = 0.0;
            }
        }
    }

    fn rebalanceFractalStructures(self: *Self) !void {
        var edge_iter = self.graph.edges.iterator();
        var total_dimension: f64 = 0.0;
        var edge_count: usize = 0;

        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_dimension += edge.fractal_dimension;
                edge_count += 1;
            }
        }

        if (edge_count > 0) {
            const avg_dimension = total_dimension / @as(f64, @floatFromInt(edge_count));
            edge_iter = self.graph.edges.iterator();
            while (edge_iter.next()) |entry| {
                for (entry.value_ptr.items) |*edge| {
                    const adjustment = (avg_dimension - edge.fractal_dimension) * 0.1;
                    edge.fractal_dimension += adjustment;
                    edge.fractal_dimension = std.math.clamp(edge.fractal_dimension, 1.0, 3.0);
                }
            }
        }
    }

    pub fn executeMetaPhase(self: *Self) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .meta, self.fast_inner_steps, self.slow_outer_steps, self.allocatePhaseId());
        errdefer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        var step: usize = 0;
        while (step < 3) : (step += 1) {
            const sub_energy = if (step % 2 == 0)
                try self.executeLocalPhaseInternal(false)
            else
                try self.executeGlobalPhaseInternal(false);

            phase.updateEnergy(sub_energy);

            if (step > 0 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;
        self.statistics.recordPhase(&phase);
        try self.phase_history.append(phase);

        return final_energy;
    }

    fn computeGraphEnergy(self: *Self) f64 {
        var total_energy: f64 = 0.0;
        var count: usize = 0;

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_energy += edge.weight * edge.fractal_dimension;
                total_energy += edge.quantum_correlation.magnitude();
                count += 1;
            }
        }

        var node_iter = self.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const cos_phase = @cos(node.phase);
            total_energy += (1.0 - cos_phase * cos_phase) / 2.0;
            count += 1;
        }

        if (count > 0) {
            return total_energy / @as(f64, @floatFromInt(count));
        }
        return 1e6;
    }

    pub fn runHierarchicalReasoning(self: *Self, max_cycles: usize) !f64 {
        const result = try self.runHierarchicalReasoningFull(max_cycles);
        return result.best_energy;
    }

    pub fn runHierarchicalReasoningFull(self: *Self, max_cycles: usize) !ReasoningResult {
        var cycle: usize = 0;
        var best_energy: f64 = std.math.inf(f64);
        var prev_combined: f64 = std.math.inf(f64);
        var total_patterns: usize = 0;

        while (cycle < max_cycles) : (cycle += 1) {
            const local_e = try self.executeLocalPhase();
            const global_e = try self.executeGlobalPhase();
            const meta_e = try self.executeMetaPhase();

            const combined = (local_e + global_e + meta_e) / 3.0;
            if (combined < best_energy) {
                best_energy = combined;
            }

            total_patterns = self.statistics.patterns_discovered;

            if (cycle > 0) {
                const delta = @abs(combined - prev_combined);
                const denom = @max(@abs(prev_combined), 1.0);
                if ((delta / denom) < 1e-6) {
                    break;
                }
            }
            prev_combined = combined;

            if (combined < 0.01) {
                break;
            }
        }

        const modulation = 1.0 / (1.0 + best_energy);

        return ReasoningResult{
            .best_energy = best_energy,
            .modulation_factor = modulation,
            .phases_completed = self.statistics.total_phases,
            .patterns_found = total_patterns,
        };
    }

    pub fn modulateTensor(self: *const Self, data: []f32, modulation: f64) void {
        if (data.len == 0) return;
        const scale: f32 = @floatCast(modulation);
        var i: usize = 0;
        while (i < data.len) : (i += 1) {
            data[i] *= scale;
        }
    }

    pub fn applyModulationToGraph(self: *Self) !f64 {
        const result = try self.runHierarchicalReasoningFull(1);
        return result.modulation_factor;
    }

    pub fn getStatistics(self: *const Self) OrchestratorStatistics {
        return self.statistics;
    }

    pub fn getPhaseHistory(self: *const Self) []const ReasoningPhase {
        return self.phase_history.items;
    }
};

test "reasoning_orchestrator_local_phase" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.2);
    try graph.addNode(n2);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 100, 12345);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    const energy = try orchestrator.executeLocalPhase();
    try std.testing.expect(std.math.isFinite(energy));
}

test "reasoning_orchestrator_global_phase" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 42);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    const energy = try orchestrator.executeGlobalPhase();
    try std.testing.expect(std.math.isFinite(energy));
}
