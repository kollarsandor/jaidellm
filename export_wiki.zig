const std = @import("std");
const mem = std.mem;
const http = std.http;
const Uri = std.Uri;

const deepwiki_url = "https://deepwiki.com/kollarsandor/jaidellm/1-jaide-v40-project-overview";
const output_file = "deepwiki_export.md";

const page_order = [_][]const u8{
    "# JAIDE v40 \xe2\x80\x94 Project Overview",
    "# Getting Started",
    "# Architecture Overview",
    "# Core Primitives",
    "# Tensor System",
    "# Memory Management",
    "# I/O and Model Persistence",
    "# Neural Processing",
    "# RSF \xe2\x80\x94 Reversible Scatter Flow Processor",
    "# OFTB \xe2\x80\x94 Orthogonal Fractal Transform Block",
    "# Tokenizer and Retrieval",
    "# MGT \xe2\x80\x94 Morpheme-Guided Tokenizer",
    "# SSI \xe2\x80\x94 Structured Sequence Index",
    "# Ranker \xe2\x80\x94 Sequence Scoring and Candidate Evaluation",
    "# NSIR \xe2\x80\x94 Quantum-Relational Graph System",
    "# NSIR Core \xe2\x80\x94 Graph Structure and Quantum Operations",
    "# Reasoning Orchestrator and Energy Minimization",
    "# CREV Pipeline \xe2\x80\x94 Knowledge Extraction and Triplet Management",
    "# Quantum Backend Integration",
    "# Optimization and Training",
    "# SFD Optimizer \xe2\x80\x94 Second-Order Training",
    "# Distributed Training",
    "# Cloud Training with Modal",
    "# Hardware Acceleration Layer",
    "# Futhark GPU Kernels",
    "# CUDA Bindings and Accelerator Interface",
    "# Clash RTL Components",
    "# Inference Server and API",
    "# InferenceServer \xe2\x80\x94 HTTP API and Request Lifecycle",
    "# Verified Inference Engine and ZK Proofs",
    "# Security, Safety, and Formal Verification",
    "# Formal Verification and Security Proofs",
    "# Safety, Obfuscation, and C API",
    "# Glossary",
};

fn unescapeJsonString(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '\\' and i + 1 < input.len) {
            switch (input[i + 1]) {
                'n' => {
                    try result.append('\n');
                    i += 2;
                },
                't' => {
                    try result.append('\t');
                    i += 2;
                },
                'r' => {
                    try result.append('\r');
                    i += 2;
                },
                '\\' => {
                    try result.append('\\');
                    i += 2;
                },
                '"' => {
                    try result.append('"');
                    i += 2;
                },
                '/' => {
                    try result.append('/');
                    i += 2;
                },
                'u' => {
                    if (i + 5 < input.len) {
                        const hex = input[i + 2 .. i + 6];
                        const codepoint = std.fmt.parseInt(u21, hex, 16) catch {
                            try result.append('\\');
                            i += 1;
                            continue;
                        };

                        if (codepoint >= 0xD800 and codepoint <= 0xDBFF) {
                            if (i + 11 < input.len and input[i + 6] == '\\' and input[i + 7] == 'u') {
                                const low_hex = input[i + 8 .. i + 12];
                                const low_surrogate = std.fmt.parseInt(u21, low_hex, 16) catch {
                                    try result.append('\\');
                                    i += 1;
                                    continue;
                                };
                                const full_codepoint: u21 = @intCast((@as(u32, codepoint - 0xD800) << 10) + (low_surrogate - 0xDC00) + 0x10000);
                                var buf: [4]u8 = undefined;
                                const len = std.unicode.utf8Encode(full_codepoint, &buf) catch {
                                    try result.append('\\');
                                    i += 1;
                                    continue;
                                };
                                try result.appendSlice(buf[0..len]);
                                i += 12;
                            } else {
                                try result.append('\\');
                                i += 1;
                            }
                        } else {
                            var buf: [4]u8 = undefined;
                            const len = std.unicode.utf8Encode(codepoint, &buf) catch {
                                try result.append('\\');
                                i += 1;
                                continue;
                            };
                            try result.appendSlice(buf[0..len]);
                            i += 6;
                        }
                    } else {
                        try result.append('\\');
                        i += 1;
                    }
                },
                else => {
                    try result.append('\\');
                    i += 1;
                },
            }
        } else {
            try result.append(input[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice();
}

fn extractMarkdownSections(allocator: std.mem.Allocator, html: []const u8) !std.ArrayList([]u8) {
    var sections = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (sections.items) |item| {
            allocator.free(item);
        }
        sections.deinit();
    }

    const marker = "self.__next_f.push([1,\"# ";
    var pos: usize = 0;

    while (pos < html.len) {
        const start = mem.indexOfPos(u8, html, pos, marker) orelse break;
        const content_start = start + "self.__next_f.push([1,\"".len;

        var end_pos = content_start;
        var depth: usize = 0;
        _ = &depth;

        while (end_pos < html.len) {
            if (html[end_pos] == '\\' and end_pos + 1 < html.len) {
                end_pos += 2;
                continue;
            }
            if (html[end_pos] == '"') {
                break;
            }
            end_pos += 1;
        }

        if (end_pos < html.len) {
            const raw_content = html[content_start..end_pos];
            const unescaped = try unescapeJsonString(allocator, raw_content);
            try sections.append(unescaped);
        }

        pos = end_pos + 1;
    }

    return sections;
}

fn orderSections(sections: *std.ArrayList([]u8)) void {
    const items = sections.items;
    var ordered_count: usize = 0;

    for (page_order) |prefix| {
        for (ordered_count..items.len) |j| {
            if (mem.startsWith(u8, items[j], prefix)) {
                if (j != ordered_count) {
                    const tmp = items[ordered_count];
                    items[ordered_count] = items[j];
                    items[j] = tmp;
                }
                ordered_count += 1;
                break;
            }
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    try stdout.print("DeepWiki Export Tool - kollarsandor/jaidellm\n", .{});
    try stdout.print("============================================\n\n", .{});
    try stdout.print("Fetching wiki content from: {s}\n\n", .{deepwiki_url});

    var client = http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try Uri.parse(deepwiki_url);

    var header_buf: [16384]u8 = undefined;
    var req = try client.open(.GET, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &.{
            .{ .name = "User-Agent", .value = "JAIDE-DeepWiki-Exporter/1.0" },
            .{ .name = "Accept", .value = "text/html" },
        },
    });
    defer req.deinit();

    try req.send();
    try req.wait();

    if (req.response.status != .ok) {
        try stdout.print("Error: HTTP {d}\n", .{@intFromEnum(req.response.status)});
        return error.HttpError;
    }

    try stdout.print("Response received, reading body...\n", .{});

    const body = try req.reader().readAllAlloc(allocator, 50 * 1024 * 1024);
    defer allocator.free(body);

    try stdout.print("HTML size: {d} bytes\n", .{body.len});
    try stdout.print("Extracting markdown sections...\n\n", .{});

    var sections = try extractMarkdownSections(allocator, body);
    defer {
        for (sections.items) |item| {
            allocator.free(item);
        }
        sections.deinit();
    }

    try stdout.print("Found {d} wiki pages\n", .{sections.items.len});

    orderSections(&sections);

    const out_file = try std.fs.cwd().createFile(output_file, .{});
    defer out_file.close();

    var writer = out_file.writer();

    try writer.writeAll("# JAIDE v40 — DeepWiki Export\n\n");
    try writer.writeAll("> Automatically exported from [DeepWiki](https://deepwiki.com/kollarsandor/jaidellm)\n\n");
    try writer.writeAll("---\n\n");

    try writer.writeAll("## Table of Contents\n\n");
    for (sections.items, 0..) |section, idx| {
        const first_newline = mem.indexOf(u8, section, "\n") orelse section.len;
        const title_line = section[0..first_newline];
        const title = if (mem.startsWith(u8, title_line, "# "))
            title_line[2..]
        else
            title_line;
        try writer.print("{d}. [{s}](#page-{d})\n", .{ idx + 1, title, idx + 1 });
    }
    try writer.writeAll("\n---\n\n");

    for (sections.items, 0..) |section, idx| {
        const first_newline = mem.indexOf(u8, section, "\n") orelse section.len;
        const title_line = section[0..first_newline];
        const title = if (mem.startsWith(u8, title_line, "# "))
            title_line[2..]
        else
            title_line;

        try writer.print("<a id=\"page-{d}\"></a>\n\n", .{idx + 1});

        try writer.writeAll(section);

        try writer.writeAll("\n\n");

        if (idx < sections.items.len - 1) {
            try writer.print("---\n\n*[Back to Table of Contents](#table-of-contents) | Page {d} of {d} | Next: ", .{ idx + 1, sections.items.len });

            const next_section = sections.items[idx + 1];
            const next_newline = mem.indexOf(u8, next_section, "\n") orelse next_section.len;
            const next_title_line = next_section[0..next_newline];
            const next_title = if (mem.startsWith(u8, next_title_line, "# "))
                next_title_line[2..]
            else
                next_title_line;
            try writer.print("{s}*\n\n", .{next_title});
        } else {
            try writer.print("---\n\n*[Back to Table of Contents](#table-of-contents) | Page {d} of {d}*\n\n", .{ idx + 1, sections.items.len });
        }

        try stdout.print("  [{d}/{d}] Exported: {s}\n", .{ idx + 1, sections.items.len, title });
    }

    try stdout.print("\nExport complete! Output written to: {s}\n", .{output_file});
    try stdout.print("Total pages exported: {d}\n", .{sections.items.len});
}
