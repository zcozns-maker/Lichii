// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader {
struct HostTranslateInfo;
}

namespace Shader::Maxwell {

[[nodiscard]] IR::Program TranslateProgram(ObjectPool<IR::Inst>& inst_pool,
                                           ObjectPool<IR::Block>& block_pool, Environment& env,
                                           Flow::CFG& cfg, const HostTranslateInfo& host_info);

[[nodiscard]] IR::Program MergeDualVertexPrograms(IR::Program& vertex_a, IR::Program& vertex_b,
                                                  Environment& env_vertex_b,
                                                  const HostTranslateInfo& host_info);

void ConvertLegacyToGeneric(IR::Program& program, const RuntimeInfo& runtime_info);

// Hosts without VK_NV_geometry_shader_passthrough lower that stage to ordinary outputs.
// A mask that also names InstanceId or VertexId would declare inputs a geometry shader
// cannot have. Keep Position and the generic varyings only.
// Ported from Eglantine PR 720 (0f9b6ab68a), GPL-3.0-or-later.
void RestrictLoweredGeometryPassthrough(VaryingState& mask);

// Drop generic geometry outputs the fragment shader does not read. Declaring all 32
// generics plus Position and Layer is 133 components. maxGeometryOutputComponents is 128
// on the AMD Windows driver that fast-fails inside vkCreateGraphicsPipelines for that
// shader (Eglantine PR 720, RX 6600) instead of returning an error. XFB, legacy, indexed
// and native passthrough programs are left unchanged. On a lowered passthrough shader the
// unread generics are copies, so their inputs are dropped too.
// Ported from Eglantine PR 720 (0f9b6ab68a), GPL-3.0-or-later.
void PruneUnreadGeometryOutputs(IR::Program& geometry, const IR::Program& fragment,
                                bool transform_feedback, bool native_passthrough);

struct GeometryInterfaceLimits {
    u32 max_output_components{};
    u32 max_total_output_components{};
    u32 max_output_vertices{};
    u32 max_invocations{};
    u32 max_input_components{};
    u32 max_clip_distances{};
    bool declare_viewport_mask{};
};

struct GeometryInterfaceFootprint {
    u32 output_components{};
    u32 input_components{};
    u32 output_vertices{};
    u32 invocations{};
    u64 total_output_components{};
    bool fits{true};
};

// Counts the varyings SPIR-V actually declares for an ordinary or lowered geometry shader.
// output_vertices and invocations of 0 are reported as 1, matching execution-mode emission.
[[nodiscard]] GeometryInterfaceFootprint MeasureGeometryInterface(
    const IR::Program& geometry, const RuntimeInfo& runtime_info, GeometryInterfaceLimits limits);

// Maxwell v1 and older Nvidia cards don't support setting gl_Layer from non-geometry stages.
// This creates a workaround by setting the layer as a generic output and creating a
// passthrough geometry shader that reads the generic and sets the layer.
[[nodiscard]] IR::Program GenerateGeometryPassthrough(ObjectPool<IR::Inst>& inst_pool,
                                                      ObjectPool<IR::Block>& block_pool,
                                                      const HostTranslateInfo& host_info,
                                                      IR::Program& source_program,
                                                      Shader::OutputTopology output_topology);

} // namespace Shader::Maxwell
