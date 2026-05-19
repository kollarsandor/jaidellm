const std = @import("std");
const mem = std.mem;
const http = std.http;
const Uri = std.Uri;
const base64 = std.base64.standard;

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

fn isSourceRefLine(line: []const u8) bool {
    const trimmed = mem.trimLeft(u8, line, " \t");
    if (mem.startsWith(u8, trimmed, "Sources:") or mem.startsWith(u8, trimmed, "**Sources:**")) return true;
    if (mem.startsWith(u8, trimmed, "---\n**Sources:**") or mem.startsWith(u8, trimmed, "---\n*Sources:*")) return true;
    if (trimmed.len > 0 and trimmed[0] == '*' and mem.indexOf(u8, trimmed, "Sources:") != null) return true;
    return false;
}

fn isOnlySourceRefs(line: []const u8) bool {
    const trimmed = mem.trimLeft(u8, line, " \t*");
    if (trimmed.len == 0) return false;
    if (mem.startsWith(u8, trimmed, "[") and mem.indexOf(u8, trimmed, "](") != null) {
        var has_non_ref = false;
        var i: usize = 0;
        while (i < trimmed.len) {
            if (trimmed[i] == '[') {
                const close = mem.indexOfPos(u8, trimmed, i, "](") orelse break;
                const paren_close = mem.indexOfPos(u8, trimmed, close, ")") orelse break;
                i = paren_close + 1;
            } else if (trimmed[i] == ' ' or trimmed[i] == ',' or trimmed[i] == '*') {
                i += 1;
            } else {
                has_non_ref = true;
                break;
            }
        }
        if (!has_non_ref) return true;
    }
    return false;
}

fn removeInlineSourceRefs(allocator: std.mem.Allocator, line: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    var i: usize = 0;
    while (i < line.len) {
        if (i + 1 < line.len and line[i] == ' ' and line[i + 1] == '[') {
            if (findSourceRef(line, i + 1)) |ref_end| {
                i = ref_end;
                continue;
            }
        }
        if (line[i] == '[') {
            if (findSourceRef(line, i)) |ref_end| {
                i = ref_end;
                continue;
            }
        }
        try result.append(line[i]);
        i += 1;
    }

    return result.toOwnedSlice();
}

fn findSourceRef(line: []const u8, start: usize) ?usize {
    if (start >= line.len or line[start] != '[') return null;
    const close_bracket = mem.indexOfPos(u8, line, start + 1, "](") orelse return null;
    const ref_text = line[start + 1 .. close_bracket];

    var has_colon = false;
    var has_dot_or_slash = false;
    for (ref_text) |c| {
        if (c == ':') has_colon = true;
        if (c == '.' or c == '/') has_dot_or_slash = true;
    }
    if (!has_dot_or_slash) return null;

    const looks_like_source = has_colon or mem.endsWith(u8, ref_text, ".zig") or
        mem.endsWith(u8, ref_text, ".fut") or mem.endsWith(u8, ref_text, ".md") or
        mem.endsWith(u8, ref_text, ".zig]") or mem.endsWith(u8, ref_text, ".toml");

    if (!looks_like_source) return null;

    const paren_close = mem.indexOfPos(u8, line, close_bracket + 2, ")") orelse return null;
    const inside_parens = line[close_bracket + 2 .. paren_close];
    if (inside_parens.len == 0 or mem.startsWith(u8, inside_parens, "http")) {
        if (inside_parens.len == 0) return paren_close + 1;
    }
    return null;
}

fn mermaidToImageUrl(allocator: std.mem.Allocator, mermaid_code: []const u8) ![]u8 {
    const encoded_len = base64.Encoder.calcSize(mermaid_code.len);
    const encoded = try allocator.alloc(u8, encoded_len);
    defer allocator.free(encoded);
    _ = base64.Encoder.encode(encoded, mermaid_code);

    const prefix = "https://mermaid.ink/img/";
    var url = try allocator.alloc(u8, prefix.len + encoded_len);
    @memcpy(url[0..prefix.len], prefix);
    @memcpy(url[prefix.len..], encoded);
    return url;
}

fn cleanMarkdown(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    var line_iter = mem.splitScalar(u8, input, '\n');
    var in_code_block = false;
    var in_mermaid_block = false;
    var in_details_block = false;
    var mermaid_buf = std.ArrayList(u8).init(allocator);
    defer mermaid_buf.deinit();
    var diagram_label: ?[]const u8 = null;
    var skip_sources_block = false;
    var first_line = true;

    while (line_iter.next()) |line| {
        if (!first_line and !in_code_block and !in_mermaid_block) {
            // will add newline before appending
        }

        if (mem.startsWith(u8, mem.trimLeft(u8, line, " "), "<details>")) {
            in_details_block = true;
            continue;
        }
        if (in_details_block) {
            if (mem.startsWith(u8, mem.trimLeft(u8, line, " "), "</details>")) {
                in_details_block = false;
            }
            continue;
        }

        if (mem.startsWith(u8, mem.trimLeft(u8, line, " "), "```")) {
            const trimmed = mem.trimLeft(u8, line, " ");
            if (in_mermaid_block) {
                in_mermaid_block = false;
                const mermaid_code = mermaid_buf.items;
                if (mermaid_code.len > 0) {
                    const url = try mermaidToImageUrl(allocator, mermaid_code);
                    defer allocator.free(url);
                    if (diagram_label) |label| {
                        if (!first_line) try result.append('\n');
                        try result.appendSlice("![");
                        try result.appendSlice(label);
                        try result.appendSlice("](");
                        try result.appendSlice(url);
                        try result.appendSlice(")");
                    } else {
                        if (!first_line) try result.append('\n');
                        try result.appendSlice("![Diagram](");
                        try result.appendSlice(url);
                        try result.appendSlice(")");
                    }
                }
                mermaid_buf.clearRetainingCapacity();
                diagram_label = null;
                first_line = false;
                continue;
            } else if (in_code_block) {
                in_code_block = false;
                first_line = false;
                continue;
            } else if (mem.startsWith(u8, trimmed[3..], "mermaid")) {
                in_mermaid_block = true;
                mermaid_buf.clearRetainingCapacity();
                first_line = false;
                continue;
            } else {
                in_code_block = true;
                first_line = false;
                continue;
            }
        }

        if (in_mermaid_block) {
            if (mermaid_buf.items.len > 0) try mermaid_buf.append('\n');
            try mermaid_buf.appendSlice(line);
            continue;
        }

        if (in_code_block) continue;

        if (mem.startsWith(u8, mem.trimLeft(u8, line, " \t*"), "**Diagram:") or
            mem.startsWith(u8, mem.trimLeft(u8, line, " \t*"), "Diagram:"))
        {
            const colon_pos = mem.indexOf(u8, line, ":") orelse 0;
            if (colon_pos + 1 < line.len) {
                const raw_label = mem.trimRight(u8, mem.trimLeft(u8, line[colon_pos + 1 ..], " "), " *");
                if (raw_label.len > 0) {
                    diagram_label = raw_label;
                }
            }
        }

        if (isSourceRefLine(line)) {
            skip_sources_block = true;
            continue;
        }

        if (skip_sources_block) {
            const trimmed = mem.trimLeft(u8, line, " \t");
            if (isOnlySourceRefs(trimmed) or trimmed.len == 0 or
                (trimmed.len > 0 and trimmed[0] == '*' and mem.indexOf(u8, trimmed, "](") != null))
            {
                continue;
            }
            skip_sources_block = false;
        }

        if (isOnlySourceRefs(mem.trimLeft(u8, line, " \t"))) continue;

        const cleaned_line = try removeInlineSourceRefs(allocator, line);
        defer allocator.free(cleaned_line);

        const final_line = mem.trimRight(u8, cleaned_line, " ");

        if (!first_line) try result.append('\n');
        try result.appendSlice(final_line);
        first_line = false;
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
    try stdout.print("Cleaning content (removing code blocks, source refs, rendering diagrams)...\n", .{});

    orderSections(&sections);

    var cleaned_sections = std.ArrayList([]u8).init(allocator);
    defer {
        for (cleaned_sections.items) |item| {
            allocator.free(item);
        }
        cleaned_sections.deinit();
    }

    for (sections.items) |section| {
        const cleaned = try cleanMarkdown(allocator, section);
        try cleaned_sections.append(cleaned);
    }

    const out_file = try std.fs.cwd().createFile(output_file, .{});
    defer out_file.close();

    var writer = out_file.writer();

    try writer.writeAll("# JAIDE v40 \xe2\x80\x94 DeepWiki Export\n\n");
    try writer.writeAll("> Automatically exported from [DeepWiki](https://deepwiki.com/kollarsandor/jaidellm)\n\n");
    try writer.writeAll("---\n\n");

    try writer.writeAll("## Table of Contents\n\n");
    for (cleaned_sections.items, 0..) |section, idx| {
        const first_newline = mem.indexOf(u8, section, "\n") orelse section.len;
        const title_line = section[0..first_newline];
        const title = if (mem.startsWith(u8, title_line, "# "))
            title_line[2..]
        else
            title_line;
        try writer.print("{d}. [{s}](#page-{d})\n", .{ idx + 1, title, idx + 1 });
    }
    try writer.writeAll("\n---\n\n");

    for (cleaned_sections.items, 0..) |section, idx| {
        const first_newline = mem.indexOf(u8, section, "\n") orelse section.len;
        const title_line = section[0..first_newline];
        const title = if (mem.startsWith(u8, title_line, "# "))
            title_line[2..]
        else
            title_line;

        try writer.print("<a id=\"page-{d}\"></a>\n\n", .{idx + 1});

        try writer.writeAll(section);

        try writer.writeAll("\n\n");

        if (idx < cleaned_sections.items.len - 1) {
            try writer.print("---\n\n*[Back to Table of Contents](#table-of-contents) | Page {d} of {d} | Next: ", .{ idx + 1, cleaned_sections.items.len });

            const next_section = cleaned_sections.items[idx + 1];
            const next_newline = mem.indexOf(u8, next_section, "\n") orelse next_section.len;
            const next_title_line = next_section[0..next_newline];
            const next_title = if (mem.startsWith(u8, next_title_line, "# "))
                next_title_line[2..]
            else
                next_title_line;
            try writer.print("{s}*\n\n", .{next_title});
        } else {
            try writer.print("---\n\n*[Back to Table of Contents](#table-of-contents) | Page {d} of {d}*\n\n", .{ idx + 1, cleaned_sections.items.len });
        }

        try stdout.print("  [{d}/{d}] Exported: {s}\n", .{ idx + 1, cleaned_sections.items.len, title });
    }

    try stdout.print("\nExport complete! Output written to: {s}\n", .{output_file});
    try stdout.print("Total pages exported: {d}\n", .{cleaned_sections.items.len});
}
