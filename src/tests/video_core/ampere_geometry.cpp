// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/object_pool.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};

bool Contains(const Shader::Ampere::LoweredProgram& lowered, u64 instruction) {
    return std::find(lowered.instructions.begin(), lowered.instructions.end(), instruction) !=
           lowered.instructions.end();
}

constexpr std::array<Instruction, 16> GEOMETRY_PROGRAM{{
    {{0x0000734e, 0, 0, 0}},
    {{0x00037802, 0xd0, 0xf00, 0}},
    {{0x00007210, 3, 0x07f1e0ff, 0}},
    {{0xff017224, 0xff, 0x000e0401, 0}},
    {{0x00007357, 0, 0, 0}},
    {{0x00007802, 0, 0xf00, 0}},
    {{0x00017802, 0, 0xf00, 0}},
    {{0x00037802, 0, 0xf00, 0}},
    {{0x0100720c, 0xff, 0x03f03070, 0}},
    {{0x00017923, 0, 0, 0}},          // ISBERD R1, [R0]
    {{0x00027324, 0xff, 0x4000, 0}},  // OUT.EMIT R2, R0, RZ
    {{0x02037924, 0, 0x8000, 0}},     // OUT.CUT R3, R2, 0
    EXIT,
    {{0x00ff7324, 0xff, 0, 0}},
    {{0x00007957, 0, 0, 0}},
    EXIT,
}};
} // namespace

TEST_CASE("Ampere geometry program counter uses the native address", "[video_core][ampere]") {
    const Instruction nop{{0x00007918, 0, 0, 0}};
    const Instruction lepc{{0x0014734e, 0, 0, 0}};
    const std::array code{nop, lepc, EXIT};
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    REQUIRE_FALSE(LowerToMaxwell(code, true));
    const auto result = LowerToMaxwell(code, {.geometry = true,
                                            .native_code_address = 0x00000012fffffff0ULL});
    REQUIRE(static_cast<bool>(result));
    REQUIRE(result.instructions.size() == 4);
    REQUIRE((result.instructions[1] & 0xff) == 20);
    REQUIRE(((result.instructions[1] >> 20) & 0xffffffffULL) == 0);
    REQUIRE((result.instructions[2] & 0xff) == 21);
    REQUIRE(((result.instructions[2] >> 20) & 0xffffffffULL) == 0x13);

    Instruction unsupported = lepc;
    unsupported.words[1] = 1;
    REQUIRE_FALSE(LowerToMaxwell(std::array{nop, unsupported, EXIT},
                                 {.geometry = true, .native_code_address = 0x00000012fffffff0ULL}));
    unsupported = lepc;
    unsupported.words[0] = (unsupported.words[0] & ~0xff0000U) | (245U << 16);
    REQUIRE_FALSE(LowerToMaxwell(std::array{nop, unsupported, EXIT},
                                 {.geometry = true, .native_code_address = 0x00000012fffffff0ULL}));
}

TEST_CASE("Ampere geometry stage map and exit setup are stage specific", "[video_core][ampere]") {
    const Instruction map_read{{0x05127923, 0, 0, 0}};
    const Instruction exit_setup{{0x03007357, 0, 0, 0}};
    auto code = GEOMETRY_PROGRAM;
    code[9] = map_read;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    const auto result = LowerToMaxwell(code, true);
    REQUIRE(static_cast<bool>(result));
    const u64 map = result.instructions[result.source_instruction_offsets[9]];
    REQUIRE((map & 0xff) == 18);
    REQUIRE(((map >> 20) & 0xff) == 5);
    REQUIRE_FALSE(LowerToMaxwell(std::array{map_read, exit_setup, EXIT}, true));

    Instruction unsupported = exit_setup;
    unsupported.words[1] = 1;
    REQUIRE_FALSE(LowerToMaxwell(std::array{unsupported, EXIT}, true));
}

TEST_CASE("Ampere F2I supports 16-bit truncation", "[video_core][ampere]") {
    for (const bool is_signed : {false, true}) {
        const Instruction convert{{0x00067305, 9, 0x0021e800U | (is_signed ? 0x100U : 0), 0}};
        const auto result = LowerToMaxwell(std::array{convert, EXIT});
        REQUIRE(static_cast<bool>(result));
        const auto word = std::ranges::find_if(result.instructions, [](u64 value) {
            return (value & 0xfff0000000000000ULL) == 0x5cb0000000000000ULL;
        });
        REQUIRE(word != result.instructions.end());
        REQUIRE(((*word >> 8) & 3) == 1);
        REQUIRE(((*word >> 12) & 1) == static_cast<u64>(is_signed));
        REQUIRE(((*word >> 39) & 3) == 3);
        REQUIRE(((*word >> 44) & 1) == 1);
    }
    const Instruction unsupported{{0x00067305, 9, 0x0021e801, 0}};
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{unsupported, EXIT})));
}

TEST_CASE("Ampere geometry programs lower their output instructions", "[video_core][ampere]") {
    const auto lowered = LowerToMaxwell(GEOMETRY_PROGRAM, true);
    INFO(lowered.error << " at " << lowered.error_offset);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.source_size == sizeof(GEOMETRY_PROGRAM));
    for (size_t index = 0; index <= Shader::Ampere::GeometryExitSetupInstructions; ++index) {
        REQUIRE(lowered.source_instruction_offsets[index] == 0);
    }
    REQUIRE(Contains(lowered, 0x5c98078000070001ULL)); // MOV R1, R0
    REQUIRE(Contains(lowered, 0xfbe000800ff70002ULL)); // OUT.EMIT R2, R0, RZ
    REQUIRE(Contains(lowered, 0xf6e0010000070203ULL)); // OUT.CUT R3, R2, 0
    REQUIRE(Contains(lowered, 0xfbe000000ff700ffULL));
    REQUIRE(lowered.instructions.back() == 0xe30000000007000fULL);
}

TEST_CASE("Ampere output instructions stay rejected outside geometry programs",
          "[video_core][ampere]") {
    const auto lowered = LowerToMaxwell(GEOMETRY_PROGRAM);
    REQUIRE(!static_cast<bool>(lowered));
    REQUIRE(lowered.error == "native program counter load outside geometry stage");
    for (const auto& output : {GEOMETRY_PROGRAM[10], GEOMETRY_PROGRAM[11]}) {
        const auto rejected = LowerToMaxwell(std::array{output, EXIT});
        REQUIRE_FALSE(rejected);
        REQUIRE(rejected.error == "native geometry output outside geometry stage");
    }
}

TEST_CASE("Ampere geometry rejects an unvalidated program counter load", "[video_core][ampere]") {
    const Instruction lepc{{0x0004734e, 0, 0, 0}};
    const auto truncated = LowerToMaxwell(std::array{lepc, EXIT}, true);
    REQUIRE_FALSE(truncated);
    REQUIRE(truncated.error == "native exit envelope truncated");
    auto invalid = GEOMETRY_PROGRAM;
    invalid[0] = lepc;
    const auto rejected = LowerToMaxwell(invalid, true);
    REQUIRE_FALSE(rejected);
    REQUIRE(rejected.error == "native exit handler installation");
}

TEST_CASE("SPIR-V geometry stream zero needs no geometry streams capability",
          "[video_core][ampere][spirv]") {
    namespace IR = Shader::IR;
    Shader::ObjectPool<IR::Inst> inst_pool;
    IR::Block block{inst_pool};
    IR::IREmitter ir{block};
    ir.EmitVertex(ir.Imm32(0));
    ir.EndPrimitive(ir.Imm32(0));
    IR::Program program;
    program.stage = Shader::Stage::Geometry;
    program.output_topology = Shader::OutputTopology::PointList;
    program.output_vertices = 1;
    program.invocations = 1;
    program.blocks.push_back(&block);
    program.post_order_blocks.push_back(&block);
    IR::AbstractSyntaxNode block_node{};
    block_node.type = IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = &block;
    IR::AbstractSyntaxNode return_node{};
    return_node.type = IR::AbstractSyntaxNode::Type::Return;
    program.syntax_list = {block_node, return_node};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;
    profile.support_geometry_streams = false;
    const auto code = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    REQUIRE(code.size() > 5);
    constexpr u32 OpCapability = 17;
    constexpr u32 OpEmitVertex = 218;
    constexpr u32 OpEndPrimitive = 219;
    constexpr u32 OpEmitStreamVertex = 220;
    constexpr u32 OpEndStreamPrimitive = 221;
    constexpr u32 CapabilityGeometry = 2;
    constexpr u32 CapabilityGeometryStreams = 54;
    u32 emits = 0, ends = 0, geometry = 0;
    for (size_t offset = 5; offset < code.size();) {
        const u32 words = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xffff;
        REQUIRE(words > 0);
        REQUIRE(offset + words <= code.size());
        if (opcode == OpCapability) {
            REQUIRE(words == 2);
            REQUIRE(code[offset + 1] != CapabilityGeometryStreams);
            geometry += code[offset + 1] == CapabilityGeometry;
        }
        emits += opcode == OpEmitVertex;
        ends += opcode == OpEndPrimitive;
        REQUIRE(opcode != OpEmitStreamVertex);
        REQUIRE(opcode != OpEndStreamPrimitive);
        offset += words;
    }
    REQUIRE(emits == 1);
    REQUIRE(ends == 1);
    REQUIRE(geometry == 1);
}

std::optional<u32> ExecutionModeLiteral(const std::vector<u32>& code, u32 mode) {
    constexpr u32 OpExecutionMode = 16;
    for (size_t offset = 5; offset < code.size();) {
        const u32 words = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xffff;
        if (words == 0 || offset + words > code.size()) {
            break;
        }
        if (opcode == OpExecutionMode && words >= 4 && code[offset + 2] == mode) {
            return code[offset + 3];
        }
        offset += words;
    }
    return std::nullopt;
}

u32 LocationCount(const std::vector<u32>& code) {
    constexpr u32 OpDecorate = 71;
    constexpr u32 Location = 30;
    u32 count = 0;
    for (size_t offset = 5; offset < code.size();) {
        const u32 words = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xffff;
        if (words == 0 || offset + words > code.size()) {
            break;
        }
        count += opcode == OpDecorate && words >= 4 && code[offset + 2] == Location;
        offset += words;
    }
    return count;
}

Shader::IR::Program MinimalGeometry(u32 output_vertices, u32 invocations) {
    Shader::IR::Program program;
    program.stage = Shader::Stage::Geometry;
    program.output_topology = Shader::OutputTopology::PointList;
    program.output_vertices = output_vertices;
    program.invocations = invocations;
    return program;
}

void FillGeneric(Shader::VaryingState& state) {
    for (size_t index = 0; index < Shader::IR::NUM_GENERICS; ++index) {
        for (size_t element = 0; element < 4; ++element) {
            state.Set(Shader::IR::Attribute::Generic0X + index * 4 + element, true);
        }
    }
}

Shader::Maxwell::GeometryInterfaceLimits DesktopGeometryLimits() {
    return {
        .max_output_components = 128,
        .max_total_output_components = 1024,
        .max_output_vertices = 256,
        .max_invocations = 32,
        .max_input_components = 128,
        .max_clip_distances = 8,
        .declare_viewport_mask = false,
    };
}

TEST_CASE("SPIR-V geometry execution modes are at least one vertex and one invocation",
          "[video_core][ampere][spirv]") {
    namespace IR = Shader::IR;
    Shader::ObjectPool<IR::Inst> inst_pool;
    IR::Block block{inst_pool};
    IR::IREmitter ir{block};
    ir.EmitVertex(ir.Imm32(0));
    IR::Program program{MinimalGeometry(0, 0)};
    program.blocks.push_back(&block);
    program.post_order_blocks.push_back(&block);
    IR::AbstractSyntaxNode block_node{};
    block_node.type = IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = &block;
    IR::AbstractSyntaxNode return_node{};
    return_node.type = IR::AbstractSyntaxNode::Type::Return;
    program.syntax_list = {block_node, return_node};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;
    const auto code = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    // SPIR-V Execution Mode: Invocations = 0, OutputVertices = 26.
    REQUIRE(ExecutionModeLiteral(code, 0) == 1);
    REQUIRE(ExecutionModeLiteral(code, 26) == 1);

    program.output_vertices = 4;
    program.invocations = 2;
    const auto kept = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    REQUIRE(ExecutionModeLiteral(kept, 0) == 2);
    REQUIRE(ExecutionModeLiteral(kept, 26) == 4);
}

TEST_CASE("Lowered geometry passthrough drops non-generic mask bits and unread outputs",
          "[video_core][ampere][spirv]") {
    namespace IR = Shader::IR;
    Shader::VaryingState mask;
    mask.Set(IR::Attribute::PositionX, true);
    mask.Set(IR::Attribute::InstanceId, true);
    mask.Set(IR::Attribute::VertexId, true);
    mask.Set(IR::Attribute::Layer, true);
    mask.Set(IR::Attribute::Generic0X, true);
    Shader::Maxwell::RestrictLoweredGeometryPassthrough(mask);
    REQUIRE(mask[IR::Attribute::PositionX]);
    REQUIRE(mask[IR::Attribute::Generic0X]);
    REQUIRE_FALSE(mask[IR::Attribute::InstanceId]);
    REQUIRE_FALSE(mask[IR::Attribute::VertexId]);
    REQUIRE_FALSE(mask[IR::Attribute::Layer]);

    IR::Program geometry;
    geometry.stage = Shader::Stage::Geometry;
    geometry.is_geometry_passthrough = true;
    geometry.output_vertices = 3;
    geometry.invocations = 1;
    FillGeneric(geometry.info.stores);
    FillGeneric(geometry.info.loads);
    FillGeneric(geometry.info.passthrough);
    geometry.info.stores.Set(IR::Attribute::PositionX, true);
    geometry.info.stores.Set(IR::Attribute::PositionY, true);
    geometry.info.stores.Set(IR::Attribute::PositionZ, true);
    geometry.info.stores.Set(IR::Attribute::PositionW, true);
    geometry.info.stores.Set(IR::Attribute::Layer, true);
    geometry.info.loads.Set(IR::Attribute::PositionX, true);
    geometry.info.loads.Set(IR::Attribute::Layer, true);
    geometry.info.passthrough.Set(IR::Attribute::PositionX, true);
    IR::Program fragment;
    fragment.stage = Shader::Stage::Fragment;
    fragment.info.loads.Set(IR::Attribute::Generic2X, true);
    fragment.info.loads.Set(IR::Attribute::Generic2Y, true);
    fragment.info.loads.Set(IR::Attribute::Generic2Z, true);
    fragment.info.loads.Set(IR::Attribute::Generic2W, true);

    const auto limits = DesktopGeometryLimits();
    Shader::RuntimeInfo runtime;
    const auto before =
        Shader::Maxwell::MeasureGeometryInterface(geometry, runtime, limits);
    REQUIRE(before.output_components == 133);
    REQUIRE(before.input_components == 133);
    REQUIRE_FALSE(before.fits);

    IR::Program feedback{geometry};
    Shader::Maxwell::PruneUnreadGeometryOutputs(feedback, fragment, true, false);
    REQUIRE(feedback.info.stores.Generic(0));
    IR::Program native{geometry};
    Shader::Maxwell::PruneUnreadGeometryOutputs(native, fragment, false, true);
    REQUIRE(native.info.stores.Generic(0));

    Shader::Maxwell::PruneUnreadGeometryOutputs(geometry, fragment, false, false);
    REQUIRE(geometry.info.stores.Generic(2));
    REQUIRE_FALSE(geometry.info.stores.Generic(0));
    REQUIRE_FALSE(geometry.info.stores.Generic(31));
    REQUIRE(geometry.info.stores[IR::Attribute::PositionX]);
    REQUIRE(geometry.info.stores[IR::Attribute::Layer]);
    REQUIRE(geometry.info.loads.Generic(2));
    REQUIRE_FALSE(geometry.info.loads.Generic(0));
    REQUIRE_FALSE(geometry.info.passthrough.Generic(0));
    const auto after = Shader::Maxwell::MeasureGeometryInterface(geometry, runtime, limits);
    REQUIRE(after.output_components == 9);
    REQUIRE(after.input_components == 9);
    REQUIRE(after.fits);
}

TEST_CASE("SPIR-V geometry store to a pruned output is omitted", "[video_core][ampere][spirv]") {
    namespace IR = Shader::IR;
    Shader::ObjectPool<IR::Inst> inst_pool;
    IR::Block block{inst_pool};
    IR::IREmitter ir{block};
    ir.SetAttribute(IR::Attribute::Generic0X, ir.Imm32(1.0f), ir.Imm32(0));
    ir.SetAttribute(IR::Attribute::Generic1X, ir.Imm32(1.0f), ir.Imm32(0));
    ir.EmitVertex(ir.Imm32(0));
    IR::Program program{MinimalGeometry(1, 1)};
    program.info.stores.Set(IR::Attribute::Generic0X, true);
    program.info.stores.Set(IR::Attribute::Generic0Y, true);
    program.info.stores.Set(IR::Attribute::Generic0Z, true);
    program.info.stores.Set(IR::Attribute::Generic0W, true);
    program.blocks.push_back(&block);
    program.post_order_blocks.push_back(&block);
    IR::AbstractSyntaxNode block_node{};
    block_node.type = IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = &block;
    IR::AbstractSyntaxNode return_node{};
    return_node.type = IR::AbstractSyntaxNode::Type::Return;
    program.syntax_list = {block_node, return_node};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;
    const auto code = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    REQUIRE(LocationCount(code) == 1);
}
