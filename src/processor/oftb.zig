const std = @import("std");
const Allocator = std.mem.Allocator;

pub const OFTB = struct {
    pub const FRACTAL_SCALE: f32 = 0.7071067811865476;

    dim: usize,

    pub fn init(d: usize) OFTB {
        std.debug.assert(d != 0);
        std.debug.assert(d <= std.math.maxInt(usize) / 2);
        return OFTB{
            .dim = d,
        };
    }

    pub fn deinit(self: *OFTB) void {
        self.* = undefined;
    }

    pub fn forwardInPlaceSlice(self: OFTB, data: []f32) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (data.len < total) return error.TensorTooSmall;
        const half = self.dim;
        const x1 = data[0..half];
        const x2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = 8;
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            x1[i..][0..VLEN].* = (va - vb) * vscale;
            x2[i..][0..VLEN].* = (va + vb) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = x1[i];
            const b = x2[i];
            x1[i] = (a - b) * scale;
            x2[i] = (a + b) * scale;
        }
    }

    pub fn inverseInPlaceSlice(self: OFTB, data: []f32) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (data.len < total) return error.TensorTooSmall;
        const half = self.dim;
        const x1 = data[0..half];
        const x2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = 8;
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            x1[i..][0..VLEN].* = (va + vb) * vscale;
            x2[i..][0..VLEN].* = (vb - va) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = x1[i];
            const b = x2[i];
            x1[i] = (a + b) * scale;
            x2[i] = (b - a) * scale;
        }
    }

    pub fn forwardInPlace(self: OFTB, x: anytype) !void {
        return self.forwardInPlaceSlice(x.data);
    }

    pub fn inverseInPlace(self: OFTB, x: anytype) !void {
        return self.inverseInPlaceSlice(x.data);
    }

    pub fn backwardInPlace(self: OFTB, grad: []f32) !void {
        return self.inverseInPlaceSlice(grad);
    }

    pub fn backwardInPlaceSlice(self: OFTB, grad: []f32) !void {
        return self.inverseInPlaceSlice(grad);
    }

    comptime {
        _ = Allocator;
    }
};

comptime {
    _ = OFTB;
}

test "OFTB forward+inverse round-trip on []f32" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    var data = try allocator.alloc(f32, dim * 2);
    defer allocator.free(data);
    var orig = try allocator.alloc(f32, dim * 2);
    defer allocator.free(orig);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        const v = (random.float(f32) - 0.5) * 4.0;
        data[i] = v;
        orig[i] = v;
    }

    var b = OFTB.init(dim);
    defer b.deinit();

    try b.forwardInPlaceSlice(data);
    try b.inverseInPlaceSlice(data);

    var max_err: f32 = 0.0;
    i = 0;
    while (i < data.len) : (i += 1) {
        const e = @abs(data[i] - orig[i]);
        if (e > max_err) max_err = e;
    }
    try std.testing.expect(max_err < 1e-4);
}

test "OFTB scale equals 1 over sqrt 2" {
    const expected: f64 = 1.0 / std.math.sqrt(2.0);
    const diff = @abs(@as(f64, @floatCast(OFTB.FRACTAL_SCALE)) - expected);
    try std.testing.expect(diff < 1e-6);
}

test "OFTB Hadamard butterfly basis vector" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    var data = try allocator.alloc(f32, dim * 2);
    defer allocator.free(data);
    @memset(data, 0.0);
    data[0] = 1.0;
    data[dim] = 1.0;

    var b = OFTB.init(dim);
    defer b.deinit();
    try b.forwardInPlaceSlice(data);

    try std.testing.expect(@abs(data[0]) < 1e-5);
    const expected_v: f32 = std.math.sqrt(2.0);
    try std.testing.expect(@abs(data[dim] - expected_v) < 1e-5);
}

test "OFTB backward equals inverse for orthogonal Hadamard" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    var grad_a = try allocator.alloc(f32, dim * 2);
    defer allocator.free(grad_a);
    var grad_b = try allocator.alloc(f32, dim * 2);
    defer allocator.free(grad_b);

    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    var i: usize = 0;
    while (i < grad_a.len) : (i += 1) {
        const v = (random.float(f32) - 0.5) * 2.0;
        grad_a[i] = v;
        grad_b[i] = v;
    }

    var b = OFTB.init(dim);
    defer b.deinit();
    try b.backwardInPlaceSlice(grad_a);
    try b.inverseInPlaceSlice(grad_b);

    i = 0;
    while (i < grad_a.len) : (i += 1) {
        try std.testing.expect(@abs(grad_a[i] - grad_b[i]) < 1e-6);
    }
}

// Audit #7 stack: simulate the "coupling layer + butterfly mixing" composition
// that RSFCore.forwardOnCore performs, confirming the whole stack is reversible
// to within machine epsilon. We approximate the affine coupling with arbitrary
// per-row scale/translation vectors and assert that an N-layer forward+inverse
// round trip recovers the original input.
test "OFTB stacked coupling+butterfly N-layer reversibility" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const dim2 = dim * 2;
    const layers: usize = 6;

    var orig = try allocator.alloc(f32, dim2);
    defer allocator.free(orig);
    var work = try allocator.alloc(f32, dim2);
    defer allocator.free(work);
    const scales = try allocator.alloc(f32, layers * dim);
    defer allocator.free(scales);
    const translations = try allocator.alloc(f32, layers * dim);
    defer allocator.free(translations);

    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const random = prng.random();
    var i: usize = 0;
    while (i < dim2) : (i += 1) {
        const v = (random.float(f32) - 0.5) * 2.0;
        orig[i] = v;
        work[i] = v;
    }
    i = 0;
    while (i < scales.len) : (i += 1) {
        // Scale must stay strictly positive and bounded for invertibility.
        scales[i] = 0.5 + random.float(f32) * 1.5;
        translations[i] = (random.float(f32) - 0.5) * 0.5;
    }

    var b = OFTB.init(dim);
    defer b.deinit();

    var l: usize = 0;
    while (l < layers) : (l += 1) {
        const x1 = work[0..dim];
        const x2 = work[dim..dim2];
        const s_row = scales[l * dim .. (l + 1) * dim];
        const t_row = translations[l * dim .. (l + 1) * dim];
        var k: usize = 0;
        while (k < dim) : (k += 1) x1[k] *= s_row[k];
        k = 0;
        while (k < dim) : (k += 1) x2[k] += t_row[k] * x1[k];
        try b.forwardInPlaceSlice(work);
    }

    var idx = layers;
    while (idx > 0) : (idx -= 1) {
        try b.inverseInPlaceSlice(work);
        const x1 = work[0..dim];
        const x2 = work[dim..dim2];
        const s_row = scales[(idx - 1) * dim .. idx * dim];
        const t_row = translations[(idx - 1) * dim .. idx * dim];
        var k: usize = 0;
        while (k < dim) : (k += 1) x2[k] -= t_row[k] * x1[k];
        k = 0;
        while (k < dim) : (k += 1) x1[k] /= s_row[k];
    }

    var max_err: f32 = 0.0;
    i = 0;
    while (i < dim2) : (i += 1) {
        const e = @abs(work[i] - orig[i]);
        if (e > max_err) max_err = e;
    }
    try std.testing.expect(max_err < 1e-4);
}
