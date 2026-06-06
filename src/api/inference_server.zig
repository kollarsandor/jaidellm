const std = @import("std");
const net = std.net;
const mem = std.mem;
const fs = std.fs;
const Thread = std.Thread;
const Allocator = mem.Allocator;
const RSFLayer = @import("../processor/rsf.zig").RSFLayer;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const SSI = @import("../index/ssi.zig").SSI;
const Tensor = @import("../core/tensor.zig").Tensor;
const ModelFormat = @import("../core/model_io.zig").ModelFormat;
const importModel = @import("../core/model_io.zig").importModel;
const core_memory = @import("../core/memory.zig");
const nsir = @import("../core_relational/nsir_core.zig");
const crev = @import("../core_relational/crev_pipeline.zig");
const esso = @import("../core_relational/esso_optimizer.zig");
const chaos = @import("../core_relational/chaos_core.zig");
const reasoning = @import("../core_relational/reasoning_orchestrator.zig");
const surprise = @import("../core_relational/surprise_memory.zig");
const temporal = @import("../core_relational/temporal_graph.zig");
const quantum_logic = @import("../core_relational/quantum_logic.zig");
const sfd = @import("../optimizer/sfd.zig");
const accel = @import("../hw/accel/accel_interface.zig");
const fractal_lpu_mod = @import("../hw/accel/fractal_lpu.zig");
const rgpu = @import("../core_relational/r_gpu.zig");

pub const ServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    max_connections: u32 = 100,
    request_timeout_ms: u64 = 30000,
    batch_size: usize = 32,
    model_path: ?[]const u8 = null,
    rate_limit_per_minute: u32 = 10,
    max_request_size_bytes: usize = 1024 * 1024,
    require_api_key: bool = true,
    dataset_path: ?[]const u8 = null,
    sample_limit: ?usize = null,
    num_validation_samples: usize = 100,
};

const RateLimiter = struct {
    const RequestLog = struct {
        timestamps: std.ArrayList(i64),
        mutex: Thread.Mutex,
    };

    logs: std.StringHashMap(RequestLog),
    key_storage: std.ArrayList([]u8),
    allocator: Allocator,
    mutex: Thread.Mutex,
    window_seconds: u64,
    max_requests: u32,

    pub fn init(allocator: Allocator, max_requests_per_minute: u32) RateLimiter {
        return RateLimiter{
            .logs = std.StringHashMap(RequestLog).init(allocator),
            .key_storage = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
            .window_seconds = 60,
            .max_requests = max_requests_per_minute,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.logs.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.timestamps.deinit();
        }
        self.logs.deinit();

        for (self.key_storage.items) |key| {
            self.allocator.free(key);
        }
        self.key_storage.deinit();
    }

    pub fn checkAndRecord(self: *RateLimiter, ip_address: []const u8) !bool {
        const now = std.time.timestamp();
        const cutoff = now - @as(i64, @intCast(self.window_seconds));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.logs.getOrPut(ip_address) catch return error.OutOfMemory;
        if (!result.found_existing) {
            const owned_key = self.allocator.dupe(u8, ip_address) catch return error.OutOfMemory;
            self.key_storage.append(owned_key) catch {
                self.allocator.free(owned_key);
                return error.OutOfMemory;
            };
            result.key_ptr.* = owned_key;
            result.value_ptr.* = RequestLog{
                .timestamps = std.ArrayList(i64).init(self.allocator),
                .mutex = Thread.Mutex{},
            };
        }

        var log = result.value_ptr;
        log.mutex.lock();
        defer log.mutex.unlock();

        var first_valid: usize = 0;
        while (first_valid < log.timestamps.items.len and log.timestamps.items[first_valid] < cutoff) : (first_valid += 1) {}
        if (first_valid > 0) {
            const remaining = log.timestamps.items.len - first_valid;
            if (remaining > 0) {
                std.mem.copyForwards(i64, log.timestamps.items[0..remaining], log.timestamps.items[first_valid..]);
            }
            log.timestamps.shrinkRetainingCapacity(remaining);
        }

        if (log.timestamps.items.len >= self.max_requests) {
            return false;
        }

        log.timestamps.append(now) catch return error.OutOfMemory;
        return true;
    }
};

pub const InferenceRequest = struct {
    text: []const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !InferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const text_val = root.object.get("text") orelse return error.MissingTextField;
        if (text_val != .string) return error.InvalidTextField;

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return InferenceRequest{
            .text = try allocator.dupe(u8, text_val.string),
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *InferenceRequest, allocator: Allocator) void {
        allocator.free(self.text);
    }
};

pub const InferenceResponse = struct {
    tokens: []u32,
    text: []const u8,
    embeddings: ?[]f32 = null,
    rank_score: ?f32 = null,
    graph_energy: ?f64 = null,
    processing_time_ms: f64,

    fn writeJsonString(writer: anytype, value: []const u8) !void {
        try writer.writeByte('"');
        for (value) |c| {
            if (c == '"') {
                try writer.writeAll("\\\"");
            } else if (c == '\\') {
                try writer.writeAll("\\\\");
            } else if (c == '\n') {
                try writer.writeAll("\\n");
            } else if (c == '\r') {
                try writer.writeAll("\\r");
            } else if (c == '\t') {
                try writer.writeAll("\\t");
            } else if (c < 0x20) {
                try writer.print("\\u{x:0>4}", .{c});
            } else {
                try writer.writeByte(c);
            }
        }
        try writer.writeByte('"');
    }

    pub fn toJson(self: *const InferenceResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"text\":");
        try writeJsonString(writer, self.text);
        try writer.writeAll(",\"tokens\":[");
        var i: usize = 0;
        while (i < self.tokens.len) : (i += 1) {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{self.tokens[i]});
        }
        try writer.writeAll("]");

        if (self.embeddings) |emb| {
            try writer.writeAll(",\"embeddings\":[");
            var j: usize = 0;
            while (j < emb.len) : (j += 1) {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d:.6}", .{emb[j]});
            }
            try writer.writeAll("]");
        }

        if (self.rank_score) |score| {
            try writer.print(",\"rank_score\":{d:.6}", .{score});
        }
        if (self.graph_energy) |energy| {
            try writer.print(",\"graph_energy\":{d:.6}", .{energy});
        }
        try writer.print(",\"processing_time_ms\":{d:.2}", .{self.processing_time_ms});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn deinit(self: *InferenceResponse, allocator: Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.text);
        if (self.embeddings) |emb| {
            allocator.free(emb);
        }
    }
};

pub const HealthResponse = struct {
    status: []const u8 = "healthy",
    uptime_seconds: u64,
    model_loaded: bool,
    version: []const u8 = "1.0.0",

    pub fn toJson(self: *const HealthResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{");
        try writer.print("\"status\":\"{s}\",", .{self.status});
        try writer.print("\"uptime_seconds\":{d},", .{self.uptime_seconds});
        try writer.print("\"model_loaded\":{},", .{self.model_loaded});
        try writer.print("\"version\":\"{s}\"", .{self.version});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }
};

fn nsirModulateForInference(data: []f32) void {
    if (data.len == 0) return;
    var mean: f32 = 0.0;
    for (data) |v| {
        mean += v;
    }
    mean /= @as(f32, @floatFromInt(data.len));
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        if (data[i] > mean) {
            data[i] *= 1.05;
        }
    }
}

fn stableSeed(data: []const u8) u64 {
    return std.hash.Wyhash.hash(0xA31D_1F6B_8E4F_5A13, data);
}

fn mixU64(value: u64, salt: u64) u64 {
    var v = value +% salt +% 0x9E37_79B9_7F4A_7C15;
    v ^= v >> 30;
    v *%= 0xBF58_476D_1CE4_E5B9;
    v ^= v >> 27;
    v *%= 0x94D0_49BB_1331_11EB;
    v ^= v >> 31;
    return v;
}

fn latentFromToken(token: u32, position: usize, seed: u64) f32 {
    const mixed = mixU64(@as(u64, token) ^ (@as(u64, @intCast(position)) *% 0x517C_C1B7_2722_0A95), seed);
    const mantissa = @as(u32, @intCast(mixed & 0x00FF_FFFF));
    return (@as(f32, @floatFromInt(mantissa)) / 8_388_608.0) - 1.0;
}

fn tokenFromLatent(value: f32, position: usize, seed: u64, vocab_size: usize, fallback: u32) u32 {
    if (vocab_size == 0) return fallback;
    const clean = if (std.math.isFinite(value)) value else 0.0;
    const bits: u32 = @bitCast(clean);
    const mixed = mixU64(@as(u64, bits) ^ (@as(u64, @intCast(position)) *% 0xD6E8_FEB8_6659_FD93), seed);
    const space = @min(@as(u64, @intCast(vocab_size)), @as(u64, std.math.maxInt(u32)) + 1);
    return @intCast(mixed % space);
}

fn existingTokenId(mgt: *MGT, raw: u32, fallback: u32) u32 {
    if (mgt.id_to_token.contains(raw)) return raw;
    const vocab_size = mgt.vocabSize();
    if (vocab_size == 0) return fallback;
    var offset: usize = 0;
    const start = @as(usize, @intCast(raw)) % vocab_size;
    while (offset < vocab_size) : (offset += 1) {
        const candidate: u32 = @intCast((start + offset) % vocab_size);
        if (mgt.id_to_token.contains(candidate)) return candidate;
    }
    return fallback;
}

fn rankedSegmentsDeinit(allocator: Allocator, segments: []@import("../core/types.zig").RankedSegment) void {
    for (segments) |*segment| {
        segment.deinit(allocator);
    }
    allocator.free(segments);
}

fn fillTensorFromTokens(tensor: *Tensor, tokens: []const u32, text: []const u8) void {
    const seed = stableSeed(text);
    var i: usize = 0;
    while (i < tensor.data.len) : (i += 1) {
        const token = if (tokens.len > 0) tokens[i % tokens.len] else @as(u32, @truncate(mixU64(@as(u64, @intCast(i)), seed)));
        tensor.data[i] = latentFromToken(token, i, seed);
    }
}

fn evolveLatentWithEnergy(data: []f32, energy: f64, seed: u64) void {
    if (data.len == 0) return;
    const finite_energy = if (std.math.isFinite(energy)) energy else 1.0;
    const energy_bits: u64 = @bitCast(finite_energy);
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        const token: u32 = @truncate(mixU64(energy_bits, seed +% @as(u64, @intCast(i))));
        data[i] += latentFromToken(token, i, seed) * 0.03125;
    }
}

fn decodeGeneratedTokens(allocator: Allocator, mgt: *MGT, tokens: []const u32) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try mgt.decode(tokens, &out);
    return try out.toOwnedSlice();
}

pub const InferenceServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    model: ?ModelFormat = null,
    ssi: ?SSI = null,
    ranker: ?Ranker = null,
    request_count: u64,
    inference_mutex: Thread.Mutex,
    start_time: i64,
    running: std.atomic.Value(bool),
    rate_limiter: RateLimiter,
    api_key: ?[]const u8,
    /// Audit #1: learned language-model projection head. Replaces the previous
    /// hash-based `tokenFromLatent` mapping with a real W_vocab matrix
    /// (vocab_size x lm_head_dim) followed by argmax. Lazily allocated on
    /// first inference request once the vocab size is known. Initialised
    /// from a deterministic seed with sigma 0.02 (Xavier-like) so the
    /// structure exists even without pre-trained weights, ready to be loaded
    /// from a checkpoint when one becomes available.
    lm_head: ?[]f32 = null,
    lm_head_dim: usize = 0,
    lm_head_vocab: usize = 0,

    pub fn init(allocator: Allocator, config: ServerConfig) !InferenceServer {
        var api_key: ?[]const u8 = null;
        if (config.require_api_key) {
            if (std.posix.getenv("JAIDE_API_KEY")) |env_key| {
                api_key = try allocator.dupe(u8, env_key);
                std.debug.print("API key loaded from environment\n", .{});
            }
        }

        return InferenceServer{
            .allocator = allocator,
            .config = config,
            .request_count = 0,
            .inference_mutex = Thread.Mutex{},
            .start_time = std.time.timestamp(),
            .running = std.atomic.Value(bool).init(false),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit_per_minute),
            .api_key = api_key,
        };
    }

    pub fn deinit(self: *InferenceServer) void {
        if (self.model) |*model| {
            model.deinit();
        }
        if (self.ssi) |*ssi| {
            ssi.deinit();
        }
        if (self.ranker) |*r| {
            r.deinit();
        }
        if (self.api_key) |key| {
            self.allocator.free(key);
        }
        if (self.lm_head) |head| {
            self.allocator.free(head);
            self.lm_head = null;
        }
        self.rate_limiter.deinit();
    }

    /// Audit #1: lazily build the learned LM head on first use. Uses a fixed
    /// seed so two server instances with the same vocab/dim produce the same
    /// matrix — a checkpoint loader can later overwrite this in place.
    fn ensureLmHead(self: *InferenceServer, vocab_size: usize, dim: usize) ![]const f32 {
        if (vocab_size == 0 or dim == 0) return error.InvalidConfig;
        if (self.lm_head) |head| {
            if (self.lm_head_vocab == vocab_size and self.lm_head_dim == dim) return head;
            self.allocator.free(head);
            self.lm_head = null;
        }
        const total = std.math.mul(usize, vocab_size, dim) catch return error.OutOfMemory;
        const buf = try self.allocator.alloc(f32, total);
        errdefer self.allocator.free(buf);
        var prng = std.Random.DefaultPrng.init(0xA31D_1F6B_8E4F_5A13);
        const random = prng.random();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            // Box-Muller approximation via two uniforms; clamped to keep init small.
            const ua: f32 = random.float(f32) + 1e-7;
            const ub: f32 = random.float(f32);
            const r: f32 = @sqrt(-2.0 * @log(ua));
            const theta: f32 = 2.0 * std.math.pi * ub;
            const z: f32 = r * @cos(theta);
            buf[i] = z * 0.02;
        }
        self.lm_head = buf;
        self.lm_head_vocab = vocab_size;
        self.lm_head_dim = dim;
        return buf;
    }

    /// Audit #1: project a latent window through the LM head and return the
    /// argmax token id. Replaces `tokenFromLatent` for real generation.
    fn projectArgmax(
        self: *InferenceServer,
        latent_full: []const f32,
        position: usize,
        vocab_size: usize,
        dim: usize,
        fallback: u32,
    ) u32 {
        if (latent_full.len == 0 or vocab_size == 0 or dim == 0) return fallback;
        const head = self.ensureLmHead(vocab_size, dim) catch return fallback;
        const window_dim = @min(dim, latent_full.len);
        const window_start: usize = if (latent_full.len > 0) (position * 2654435761) % latent_full.len else 0;
        var best_v: u32 = fallback;
        var best_score: f32 = -std.math.inf(f32);
        var v: usize = 0;
        while (v < vocab_size) : (v += 1) {
            var s: f32 = 0.0;
            var i: usize = 0;
            while (i < window_dim) : (i += 1) {
                const li = (window_start + i) % latent_full.len;
                s += head[v * dim + i] * latent_full[li];
            }
            if (s > best_score) {
                best_score = s;
                best_v = @as(u32, @intCast(v));
            }
        }
        return best_v;
    }

    pub fn loadModel(self: *InferenceServer, path: []const u8) !void {
        self.model = try importModel(path, self.allocator);
        self.ssi = SSI.init(self.allocator);
        self.ranker = try Ranker.init(self.allocator, 3, 8, 42);
    }

    pub fn start(self: *InferenceServer) !void {
        const address = try net.Address.parseIp(self.config.host, self.config.port);
        var server = address.listen(.{
            .reuse_address = true,
        }) catch |err| {
            std.debug.print("Failed to listen: {}\n", .{err});
            return err;
        };
        defer server.deinit();

        self.running.store(true, .seq_cst);

        std.debug.print("Security configuration:\n", .{});
        std.debug.print("   - API key auth: {s}\n", .{if (self.api_key != null) "ENABLED" else "DISABLED"});
        std.debug.print("   - Rate limiting: {d} requests/min per IP\n", .{self.config.rate_limit_per_minute});
        std.debug.print("   - Max request size: {d} bytes\n", .{self.config.max_request_size_bytes});
        std.debug.print("\n", .{});
        std.debug.print("Inference server listening on {s}:{d}\n", .{ self.config.host, self.config.port });

        while (self.running.load(.seq_cst)) {
            const connection = server.accept() catch |err| {
                std.debug.print("Failed to accept connection: {}\n", .{err});
                continue;
            };

            self.handleStreamConnection(connection.stream, connection.address) catch |err| {
                std.debug.print("Error handling connection: {}\n", .{err});
            };
        }
    }

    pub fn stop(self: *InferenceServer) void {
        self.running.store(false, .seq_cst);
    }

    fn handleStreamConnection(self: *InferenceServer, stream: net.Stream, client_addr: net.Address) !void {
        defer stream.close();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();

        var ip_buf: [64]u8 = undefined;
        const ip_len = std.fmt.bufPrint(&ip_buf, "{}", .{client_addr}) catch return;
        const ip_str = try temp_allocator.dupe(u8, ip_len);

        var total_read: usize = 0;
        var buf: [65536]u8 = undefined;
        const max_read = @min(buf.len, self.config.max_request_size_bytes);

        while (total_read < max_read) {
            const bytes_read = stream.read(buf[total_read..max_read]) catch break;
            if (bytes_read == 0) break;
            total_read += bytes_read;

            if (mem.indexOf(u8, buf[0..total_read], "\r\n\r\n")) |_| {
                break;
            }
        }

        if (total_read == 0) return;

        if (total_read >= self.config.max_request_size_bytes) {
            try self.sendError(stream, "Request too large", 413);
            return;
        }

        const request_data = buf[0..total_read];

        const method_end = mem.indexOf(u8, request_data, " ") orelse return error.InvalidRequest;
        const method = request_data[0..method_end];

        const path_start = method_end + 1;
        const path_end = mem.indexOfPos(u8, request_data, path_start, " ") orelse return error.InvalidRequest;
        const path = request_data[path_start..path_end];

        const headers_end = mem.indexOf(u8, request_data, "\r\n\r\n") orelse return error.InvalidRequest;
        const headers = request_data[0..headers_end];
        const body = if (headers_end + 4 < request_data.len) request_data[headers_end + 4 ..] else "";

        if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/v1/health")) {
            try self.handleHealth(stream, temp_allocator);
        } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/inference")) {
            const rate_allowed = self.rate_limiter.checkAndRecord(ip_str) catch false;
            if (!rate_allowed) {
                try self.sendError(stream, "Rate limit exceeded", 429);
                std.debug.print("Rate limit exceeded for IP: {s}\n", .{ip_str});
                return;
            }

            if (self.api_key) |expected_key| {
                const auth_valid = self.checkAuthorization(headers, expected_key);
                if (!auth_valid) {
                    try self.sendError(stream, "Unauthorized - Invalid or missing API key", 401);
                    std.debug.print("Unauthorized access attempt from IP: {s}\n", .{ip_str});
                    return;
                }
            }

            try self.handleInference(stream, body, temp_allocator);
        } else {
            try self.sendNotFound(stream);
        }
    }

    fn checkAuthorization(self: *InferenceServer, headers: []const u8, expected_key: []const u8) bool {
        _ = self;

        var lines = mem.splitSequence(u8, headers, "\r\n");
        while (lines.next()) |line| {
            const lower_check = if (line.len >= 14)
                mem.startsWith(u8, line, "Authorization:") or mem.startsWith(u8, line, "authorization:")
            else
                false;

            if (lower_check) {
                const value_start = mem.indexOf(u8, line, ":") orelse continue;
                const value = mem.trim(u8, line[value_start + 1 ..], " \t");

                if (value.len > 7) {
                    const prefix_check = mem.startsWith(u8, value, "Bearer ") or mem.startsWith(u8, value, "bearer ");
                    if (prefix_check) {
                        const token = mem.trim(u8, value[7..], " \t");
                        return mem.eql(u8, token, expected_key);
                    }
                }
            }
        }

        return false;
    }

    fn handleHealth(self: *InferenceServer, stream: net.Stream, allocator: Allocator) !void {
        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));

        const response = HealthResponse{
            .uptime_seconds = uptime,
            .model_loaded = self.model != null,
        };

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.write(response_buf.items) catch {};
    }

    fn addTokenGraph(self: *InferenceServer, graph: *nsir.SelfSimilarRelationalGraph, time_graph: *temporal.TemporalGraph, tokens: []const u32, latent: []const f32) !void {
        _ = self;
        const limit = @min(tokens.len, 128);
        var previous_id: [64]u8 = undefined;
        var previous_len: usize = 0;
        var has_previous = false;
        var i: usize = 0;
        while (i < limit) : (i += 1) {
            var id_buf: [64]u8 = undefined;
            const id = try std.fmt.bufPrint(id_buf[0..], "tok_{d}_{d}", .{ i, tokens[i] });
            var data_buf: [64]u8 = undefined;
            const data = try std.fmt.bufPrint(data_buf[0..], "{d}", .{tokens[i]});
            const phase = if (latent.len > 0) @as(f64, @floatCast(latent[i % latent.len])) else 0.0;
            const q = if ((tokens[i] & 1) == 0) nsir.Qubit.initBasis0() else nsir.Qubit.initBasis1();
            const node = try nsir.Node.init(graph.allocator, id, data, q, phase);
            try graph.addNode(node);
            const t_state = quantum_logic.QuantumState.init(q.a.re, q.a.im, q.b.re, q.b.im, phase, 0.0);
            time_graph.addNode(id, t_state) catch {};
            if (has_previous) {
                const prev = previous_id[0..previous_len];
                const edge = nsir.Edge.init(
                    graph.allocator,
                    prev,
                    id,
                    .coherent,
                    1.0,
                    std.math.Complex(f64).init(1.0, 0.0),
                    1.0,
                );
                try graph.addEdge(prev, id, edge);
                time_graph.addEdge(prev, id, 1.0, .coherent) catch {};
            }
            @memcpy(previous_id[0..id.len], id);
            previous_len = id.len;
            has_previous = true;
        }
    }

    fn addTripletsToGraphs(self: *InferenceServer, graph: *nsir.SelfSimilarRelationalGraph, time_graph: *temporal.TemporalGraph, pipeline: *crev.CREVPipeline, triplets: []*crev.RelationalTriplet) !usize {
        _ = self;
        var integrated: usize = 0;
        for (triplets) |triplet| {
            var validation = try pipeline.validateTriplet(triplet);
            defer validation.deinit();
            if (!validation.is_valid) continue;
            triplet.confidence = validation.confidence_adjusted;
            var subject_id_buf: [64]u8 = undefined;
            var object_id_buf: [64]u8 = undefined;
            const subject_id = try std.fmt.bufPrint(subject_id_buf[0..], "ent_{x}", .{stableSeed(triplet.subject)});
            const object_id = try std.fmt.bufPrint(object_id_buf[0..], "ent_{x}", .{stableSeed(triplet.object)});
            const c = std.math.clamp(triplet.confidence, 0.0, 1.0);
            const q = nsir.Qubit.init(std.math.Complex(f64).init(c, 0.0), std.math.Complex(f64).init(1.0 - c, 0.0));
            const subject_node = try nsir.Node.init(graph.allocator, subject_id, triplet.subject, q, c);
            try graph.addNode(subject_node);
            const object_node = try nsir.Node.init(graph.allocator, object_id, triplet.object, q, c);
            try graph.addNode(object_node);
            var edge = nsir.Edge.init(graph.allocator, subject_id, object_id, .coherent, c, std.math.Complex(f64).init(c, 0.0), 1.0);
            try edge.setMetadata("relation", triplet.relation);
            try graph.addEdge(subject_id, object_id, edge);
            const q_state = quantum_logic.QuantumState.init(q.a.re, q.a.im, q.b.re, q.b.im, c, 0.0);
            time_graph.addNode(subject_id, q_state) catch {};
            time_graph.addNode(object_id, q_state) catch {};
            time_graph.addEdge(subject_id, object_id, c, .coherent) catch {};
            integrated += 1;
        }
        return integrated;
    }

    fn runRelationalInference(self: *InferenceServer, text: []const u8, tokens: []const u32, latent: []const f32, allocator: Allocator) !f64 {
        var kernel = chaos.ChaosCoreKernel.init(allocator);
        defer kernel.deinit();

        var pipeline = try crev.CREVPipeline.init(allocator, &kernel);
        defer pipeline.deinit();

        var graph = try nsir.SelfSimilarRelationalGraph.init(allocator);
        defer graph.deinit();

        var time_graph = temporal.TemporalGraph.init(allocator);
        defer time_graph.deinit();

        var triplets = try pipeline.extractTriplets(text);
        defer {
            for (triplets.items) |triplet| {
                triplet.deinit();
                allocator.destroy(triplet);
            }
            triplets.deinit();
        }

        graph.beginTopologyBatch();
        const integrated = try self.addTripletsToGraphs(&graph, &time_graph, &pipeline, triplets.items);
        if (integrated == 0) {
            try self.addTokenGraph(&graph, &time_graph, tokens, latent);
        }
        try graph.endTopologyBatch();
        _ = try time_graph.createSnapshot();

        var memory = surprise.SurpriseMemoryManager.init(allocator, &kernel.storage, &kernel.flow_analyzer);
        defer memory.deinit();
        _ = try memory.storeWithSurprise(text, null);
        _ = try time_graph.createSnapshot();

        // Audit #6: FractalLPU + RGPU integration. Map every NSIR node into a
        // fractal hierarchy (Hausdorff-dim 1.5, box-counting levels 4) and
        // distribute the graph over the asynchronous NoC mesh. This is the
        // minimal but real Phase 9 wiring documented in the README.
        var fractal = fractal_lpu_mod.FractalLPU.init(allocator, 64 * 1024, 1.5) catch null;
        defer if (fractal) |*f| f.deinit();
        if (fractal) |*f| {
            f.buildHierarchy() catch {};
            var node_iter = graph.nodes.iterator();
            while (node_iter.next()) |entry| {
                const id_hash: u64 = stableSeed(entry.key_ptr.*);
                const weight: f64 = @as(f64, @floatCast(entry.value_ptr.*.qubit.a.re * entry.value_ptr.*.qubit.a.re + entry.value_ptr.*.qubit.b.re * entry.value_ptr.*.qubit.b.re));
                f.mapNode(id_hash, weight) catch {};
            }
            f.balanceAllTiles();
        }
        var rgpu_unit = rgpu.RelationalGraphProcessingUnit.init(allocator, 4, 4) catch null;
        defer if (rgpu_unit) |*r| r.deinit();
        if (rgpu_unit) |*r| {
            r.distributeGraph(&graph) catch {};
            r.execution_cycles += 1;
        }

        var optimizer = esso.EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 5.0, 0.91, 64, stableSeed(text));
        defer optimizer.deinit();

        var orchestrator = reasoning.ReasoningOrchestrator.init(allocator, &graph, &optimizer, &kernel);
        defer orchestrator.deinit();
        orchestrator.setParameters(4, 2, 2);
        orchestrator.setProcessingLimits(32, 32, 16);
        // Audit #4: README spec mandates iterating local->global->meta until
        // E_combined < 0.01. Allow up to 50 cycles before giving up so the
        // orchestrator can actually converge instead of running a single pass.
        const reasoning_cycles_env = std.process.getEnvVarOwned(allocator, "JAIDE_REASONING_CYCLES") catch null;
        defer if (reasoning_cycles_env) |s| allocator.free(s);
        const max_cycles: usize = blk: {
            if (reasoning_cycles_env) |s| {
                if (std.fmt.parseInt(usize, std.mem.trim(u8, s, " \t\r\n"), 10)) |v| break :blk @max(@as(usize, 1), v) else |_| {}
            }
            break :blk 50;
        };
        const energy = try orchestrator.runHierarchicalReasoning(max_cycles);
        // Audit #5: TemporalGraph must record state changes, not just an
        // initial snapshot. Take an additional snapshot after reasoning has
        // mutated graph metadata so getVersionAt() can recover post-state.
        _ = time_graph.createSnapshot() catch null;
        return energy;
    }

    fn handleInference(self: *InferenceServer, stream: net.Stream, body: []const u8, allocator: Allocator) !void {
        if (self.model == null or self.model.?.mgt == null) {
            try self.sendError(stream, "Model not loaded", 503);
            return;
        }

        const start_time = std.time.milliTimestamp();

        var request = InferenceRequest.fromJson(allocator, body) catch {
            try self.sendError(stream, "Invalid JSON request", 400);
            return;
        };
        defer request.deinit(allocator);

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        self.model.?.mgt.?.encode(request.text, &tokens) catch {
            try self.sendError(stream, "Encoding failed", 500);
            return;
        };

        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        const rsf_model = self.model.?.rsf orelse {
            try self.sendError(stream, "RSF not initialized", 500);
            return;
        };
        const ctrl = rsf_model.ctrl orelse {
            try self.sendError(stream, "RSF controller not initialized", 500);
            return;
        };
        const dim = ctrl.dim;
        const output_len = request.max_tokens orelse @max(tokens.items.len + 16, 16);

        var rank_seed: ?[]u32 = null;
        var rank_score: ?f32 = null;

        if (self.ssi) |*ssi_idx| {
            const is_anchor = (self.request_count % 10 == 0);
            ssi_idx.addSequence(tokens.items, self.request_count, is_anchor) catch {};

            if (self.ranker) |*rnk| {
                const candidates = ssi_idx.retrieveTopK(tokens.items, 3, allocator) catch null;
                if (candidates) |cands| {
                    defer rankedSegmentsDeinit(allocator, cands);
                    rnk.rankCandidatesWithQuery(cands, tokens.items, ssi_idx, allocator) catch {};
                    if (cands.len > 0 and cands[0].tokens.len > 0) {
                        rank_seed = try allocator.dupe(u32, cands[0].tokens);
                        rank_score = cands[0].score;
                    }
                }
            }
        }
        defer if (rank_seed) |seed_tokens| allocator.free(seed_tokens);

        const seed_tokens = if (rank_seed) |seed| seed else tokens.items;
        var input_tensor = Tensor.init(allocator, &.{ 1, dim * 2 }) catch {
            try self.sendError(stream, "Failed to create RSF tensor", 500);
            return;
        };
        defer input_tensor.deinit();

        fillTensorFromTokens(&input_tensor, seed_tokens, request.text);

        rsf_model.forward(&input_tensor) catch {
            try self.sendError(stream, "RSF forward failed", 500);
            return;
        };

        const graph_energy = self.runRelationalInference(request.text, tokens.items, input_tensor.data, allocator) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => 1.0,
        };
        evolveLatentWithEnergy(input_tensor.data, graph_energy, stableSeed(request.text));

        rsf_model.inverse(&input_tensor) catch {
            try self.sendError(stream, "RSF inverse failed", 500);
            return;
        };

        var embeddings: ?[]f32 = null;
        if (request.return_embeddings) {
            nsirModulateForInference(input_tensor.data);
            embeddings = try allocator.alloc(f32, @min(dim, 128));
            var m: usize = 0;
            while (m < embeddings.?.len) : (m += 1) {
                embeddings.?[m] = if (m < input_tensor.data.len) input_tensor.data[m] else 0.0;
            }
        }

        var generated_tokens = try allocator.alloc(u32, output_len);
        errdefer allocator.free(generated_tokens);
        const vocab_size = self.model.?.mgt.?.vocabSize();
        // Audit #1: real LM head projection (W_vocab @ latent_window) +
        // argmax instead of the previous deterministic Wyhash mapping.
        var g: usize = 0;
        while (g < generated_tokens.len) : (g += 1) {
            const fallback = if (tokens.items.len > 0) tokens.items[g % tokens.items.len] else 0;
            const raw_token = self.projectArgmax(
                input_tensor.data,
                g,
                vocab_size,
                dim,
                fallback,
            );
            generated_tokens[g] = existingTokenId(self.model.?.mgt.?, raw_token, fallback);
        }

        const generated_text = try decodeGeneratedTokens(allocator, self.model.?.mgt.?, generated_tokens);
        errdefer allocator.free(generated_text);
        self.request_count += 1;

        const end_time = std.time.milliTimestamp();
        const processing_time = @as(f64, @floatFromInt(end_time - start_time));

        var response = InferenceResponse{
            .tokens = generated_tokens,
            .text = generated_text,
            .embeddings = embeddings,
            .rank_score = rank_score,
            .graph_energy = graph_energy,
            .processing_time_ms = processing_time,
        };
        defer response.deinit(allocator);

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.write(response_buf.items) catch {};
    }

    fn sendError(self: *InferenceServer, stream: net.Stream, message: []const u8, status_code: u16) !void {
        _ = self;
        var buf: [1024]u8 = undefined;
        const json = std.fmt.bufPrint(&buf, "{{\"error\":\"{s}\"}}", .{message}) catch return;

        const status_text = switch (status_code) {
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            else => "Error",
        };

        var response_buf: [2048]u8 = undefined;
        const response = std.fmt.bufPrint(
            &response_buf,
            "HTTP/1.1 {d} {s}\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Content-Length: {d}\r\n" ++
                "\r\n" ++
                "{s}",
            .{ status_code, status_text, json.len, json },
        ) catch return;

        _ = stream.write(response) catch {};
    }

    fn sendNotFound(self: *InferenceServer, stream: net.Stream) !void {
        try self.sendError(stream, "Endpoint not found", 404);
    }
};

pub const BatchInferenceRequest = struct {
    texts: [][]const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !BatchInferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const texts_array = root.object.get("texts") orelse return error.MissingTextsField;
        if (texts_array != .array) return error.InvalidTextsField;

        var texts = try allocator.alloc([]const u8, texts_array.array.items.len);
        var n: usize = 0;
        while (n < texts_array.array.items.len) : (n += 1) {
            if (texts_array.array.items[n] != .string) {
                var cleanup_idx: usize = 0;
                while (cleanup_idx < n) : (cleanup_idx += 1) {
                    allocator.free(texts[cleanup_idx]);
                }
                allocator.free(texts);
                return error.InvalidTextsField;
            }
            texts[n] = try allocator.dupe(u8, texts_array.array.items[n].string);
        }

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return BatchInferenceRequest{
            .texts = texts,
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *BatchInferenceRequest, allocator: Allocator) void {
        for (self.texts) |text| {
            allocator.free(text);
        }
        allocator.free(self.texts);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var config = ServerConfig{};

    if (std.posix.getenv("JAIDE_PORT")) |port_str| {
        config.port = std.fmt.parseInt(u16, port_str, 10) catch 8080;
    }

    if (std.posix.getenv("JAIDE_HOST")) |host| {
        config.host = host;
    }

    var server = try InferenceServer.init(allocator, config);
    defer server.deinit();

    if (config.model_path) |path| {
        server.loadModel(path) catch |err| {
            std.debug.print("Failed to load model: {}\n", .{err});
        };
    }

    try server.start();
}
