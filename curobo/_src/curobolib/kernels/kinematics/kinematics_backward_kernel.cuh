/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/math.cuh"
#include "common/pose_util.cuh"
#include "common/quaternion_util.cuh"

#include "kinematics_constants.h"
#include "third_party/helper_math.h"
#include "kinematics_util.cuh"
#include "kinematics_jacobian_backward_helper.cuh"

namespace curobo{
namespace kinematics{

    // Jacobian-only VJP. Each lane owns output joints and writes them directly;
    // no per-thread all-joint accumulator or warp reduction is required.
    template<
      typename AccumulatorType,
      int NumLinks,
      int NumJoints,
      int NumToolFrames,
      bool Accumulate>
    __global__ void kinematics_jacobian_backward_kernel(
      float *grad_out_link_q,
      const float *grad_jacobian,
      const float *global_cumul_mat,
      const int8_t *joint_map_type,
      const int16_t *tool_joint_links_data,
      const int16_t *tool_joint_links_offsets,
      const uint32_t *joint_link_influence_words,
      const float *joint_offset,
      const int16_t *tool_frame_map,
      const int batch_size)
    {
      static_assert(NumLinks > 0, "NumLinks must be positive");
      static_assert(NumJoints > 0, "NumJoints must be positive");
      static_assert(NumToolFrames > 0, "NumToolFrames must be positive");
      constexpr int ThreadsPerBatch = 32;

      extern __shared__ __align__(16) float cumul_mat[];
      const int global_thread = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = global_thread / ThreadsPerBatch;
      const int element_idx = threadIdx.x % ThreadsPerBatch;
      const int local_batch = threadIdx.x / ThreadsPerBatch;
      const int batches_per_block = blockDim.x / ThreadsPerBatch;
      const int start_batch_index = blockIdx.x * batches_per_block;
      const int matrix_address_base = local_batch * NumLinks * 12;

      read_cumul_to_shared_mem(
        cumul_mat,
        global_cumul_mat,
        start_batch_index,
        threadIdx.x,
        NumLinks,
        batch_size,
        batches_per_block,
        ThreadsPerBatch);

      if (batch >= batch_size) {
        return;
      }
      compute_jacobian_derivatives<AccumulatorType, Accumulate>(
        &grad_out_link_q[batch * NumJoints],
        grad_jacobian,
        cumul_mat,
        joint_map_type,
        tool_joint_links_data,
        tool_joint_links_offsets,
        joint_link_influence_words,
        joint_offset,
        tool_frame_map,
        batch,
        NumJoints,
        NumLinks,
        NumToolFrames,
        matrix_address_base,
        element_idx,
        ThreadsPerBatch);
    }

    // Standard-output backward. Local output cotangents form world-frame link
    // wrenches in shared memory, then one warp projects descendant wrenches onto
    // the robot joints without a global intermediate or a second kernel launch.
    template<
      typename ScalarType,
      int NumLinks,
      int NumJoints,
      int NumSpheres,
      int NumToolFrames,
      bool ComputeCom,
      int ThreadsPerBlock,
      bool SingleLinkPerJoint>
    __global__ void kinematics_link_wrench_backward_kernel(
      float *grad_out_link_q,
      const float *grad_nlinks_pos,
      const float *grad_nlinks_quat,
      const ScalarType *grad_spheres,
      const ScalarType *grad_spheres_addend,
      const float *grad_center_of_mass,
      const float *batch_center_of_mass,
      const float *global_cumul_mat,
      const float *robot_spheres,
      const float *link_masses_com,
      const int8_t *joint_map_type,
      const uint32_t *joint_link_influence_words,
      const int16_t *joint_links_data,
      const int16_t *joint_links_offsets,
      const float *joint_offset,
      const int16_t *tool_frame_map,
      const int16_t *link_sphere_map,
      const int32_t *env_query_idx,
      const int batch_size,
      const int horizon,
      const int num_envs)
    {
      static_assert(NumLinks > 0, "NumLinks must be positive");
      static_assert(NumJoints > 0, "NumJoints must be positive");
      static_assert(NumSpheres >= 0, "NumSpheres must be non-negative");
      static_assert(NumToolFrames >= 0, "NumToolFrames must be non-negative");
      static_assert(
        ThreadsPerBlock >= 32 && ThreadsPerBlock <= 128 && ThreadsPerBlock % 32 == 0,
        "The link-wrench kernel requires 32 to 128 threads in whole warps");

      const int batch = blockIdx.x;
      if (batch >= batch_size) {
        return;
      }

      constexpr int WRENCH_STRIDE = 8;
      constexpr int MATRIX_STRIDE = 12;
      extern __shared__ __align__(16) float shared_data[];
      float *shared_wrench = shared_data;
      float *cumul_mat = &shared_data[NumLinks * WRENCH_STRIDE];
      const int batch_matrix_offset = batch * NumLinks * MATRIX_STRIDE;

      for (int index = threadIdx.x; index < NumLinks * WRENCH_STRIDE;
           index += ThreadsPerBlock) {
        shared_wrench[index] = 0.0f;
      }
      // [start, end) of each link's spheres when link_sphere_map is sorted.
      __shared__ int link_sphere_range[2 * NumLinks];
      for (int index = threadIdx.x; index < 2 * NumLinks; index += ThreadsPerBlock) {
        link_sphere_range[index] = 0;
      }
      for (int index = threadIdx.x; index < NumLinks * MATRIX_STRIDE;
           index += ThreadsPerBlock) {
        cumul_mat[index] = global_cumul_mat[batch_matrix_offset + index];
      }
      __syncthreads();

      // DETERMINISTIC LINK-WRENCH ACCUMULATION, two stages.
      // Float atomicAdd into the link wrench made the addition order depend
      // on warp scheduling (joint gradients differed by ~1 ULP between
      // identical launches). Stage 1 keeps the transforms parallel and writes
      // each sphere/tool-frame contribution to its own shared slot; stage 2
      // sums each link's slots in ascending index order (one thread per link).
      constexpr int CONTRIB_STRIDE = 6;
      float *contrib = &cumul_mat[NumLinks * MATRIX_STRIDE];
      if constexpr (NumSpheres > 0) {
        const int env_idx = num_envs > 1 ? env_query_idx[batch / horizon] : 0;
        const int sphere_config_offset = env_idx * NumSpheres * 4;
        const int batch_sphere_offset = batch * NumSpheres * 4;

        for (int sphere_idx = threadIdx.x; sphere_idx < NumSpheres;
             sphere_idx += ThreadsPerBlock) {
          float *slot = &contrib[sphere_idx * CONTRIB_STRIDE];
          const int sphere_offset = sphere_idx * 4;
          float4 sphere_grad =
            *(const float4 *)&grad_spheres[batch_sphere_offset + sphere_offset];
          if (grad_spheres_addend != nullptr) {
            const float4 addend = *(const float4 *)&grad_spheres_addend[
              batch_sphere_offset + sphere_offset];
            sphere_grad.x += addend.x;
            sphere_grad.y += addend.y;
            sphere_grad.z += addend.z;
          }
          if (sphere_grad.x == 0.0f && sphere_grad.y == 0.0f &&
              sphere_grad.z == 0.0f) {
            slot[0] = 0.0f; slot[1] = 0.0f; slot[2] = 0.0f;
            slot[3] = 0.0f; slot[4] = 0.0f; slot[5] = 0.0f;
            continue;
          }

          const int link_idx = link_sphere_map[sphere_idx];
          float4 sphere_world;
          transform_sphere_float4(
            &global_cumul_mat[batch_matrix_offset + link_idx * MATRIX_STRIDE],
            &robot_spheres[sphere_config_offset + sphere_offset],
            sphere_world);
          const float3 force = make_float3(sphere_grad);
          const float3 moment = cross(make_float3(sphere_world), force);
          slot[0] = moment.x; slot[1] = moment.y; slot[2] = moment.z;
          slot[3] = force.x;  slot[4] = force.y;  slot[5] = force.z;
        }
      }

      const int batch_tool_offset = batch * NumToolFrames;
      for (int tool_idx = threadIdx.x; tool_idx < NumToolFrames;
           tool_idx += ThreadsPerBlock) {
        float *slot = &contrib[(NumSpheres + tool_idx) * CONTRIB_STRIDE];
        const float3 position_grad = *(const float3 *)&grad_nlinks_pos[
          (batch_tool_offset + tool_idx) * 3];
        __align__(16) float quaternion_grad[4];
        *(float4 *)&quaternion_grad[0] = *(const float4 *)&grad_nlinks_quat[
          (batch_tool_offset + tool_idx) * 4];
        if (position_grad.x == 0.0f && position_grad.y == 0.0f &&
            position_grad.z == 0.0f && quaternion_grad[0] == 0.0f &&
            quaternion_grad[1] == 0.0f && quaternion_grad[2] == 0.0f &&
            quaternion_grad[3] == 0.0f) {
          slot[0] = 0.0f; slot[1] = 0.0f; slot[2] = 0.0f;
          slot[3] = 0.0f; slot[4] = 0.0f; slot[5] = 0.0f;
          continue;
        }

        const int link_idx = tool_frame_map[tool_idx];
        const common::CuPose pose = common::CuPose::from_transform_matrix(
          &global_cumul_mat[batch_matrix_offset + link_idx * MATRIX_STRIDE]);
        float3 orientation_grad = make_float3(0.0f, 0.0f, 0.0f);
        common::quaternion_cotangent_to_angular_cotangent(
          pose.quaternion, &quaternion_grad[0], orientation_grad);
        const float3 moment = cross(pose.position, position_grad) + orientation_grad;
        slot[0] = moment.x; slot[1] = moment.y; slot[2] = moment.z;
        slot[3] = position_grad.x; slot[4] = position_grad.y; slot[5] = position_grad.z;
      }
      // Mark each link's contiguous sphere range (unique writer per entry when
      // the map is sorted) and detect an unsorted map block-wide; the OR is
      // also the barrier between the two stages.
      int map_unsorted = 0;
      if constexpr (NumSpheres > 0) {
        for (int sphere_idx = threadIdx.x; sphere_idx < NumSpheres;
             sphere_idx += ThreadsPerBlock) {
          const int link = link_sphere_map[sphere_idx];
          const int prev = sphere_idx > 0 ? link_sphere_map[sphere_idx - 1] : -1;
          const int next = sphere_idx + 1 < NumSpheres ? link_sphere_map[sphere_idx + 1] : -1;
          if (sphere_idx > 0 && prev > link) {
            map_unsorted = 1;
          }
          if (link >= 0 && link < NumLinks) {
            if (prev != link) {
              link_sphere_range[2 * link] = sphere_idx;
            }
            if (next != link) {
              link_sphere_range[2 * link + 1] = sphere_idx + 1;
            }
          }
        }
      }
      const bool sorted_map = __syncthreads_or(map_unsorted) == 0;

      // Stage 2: LINK_GROUP threads per link. Thread j of a group sums slots
      // first+j, first+j+LINK_GROUP, ... in ascending order (spheres, then tool
      // frames); a fixed width-LINK_GROUP xor butterfly combines the group and
      // its lane 0 writes the wrench. Fixed order everywhere: deterministic.
      // This shortens the dependent add chain of large links (latency-bound
      // kernel) compared with one thread summing a whole link.
      {
        constexpr int LINK_GROUP = 8;
        constexpr int GroupsPerBlock = ThreadsPerBlock / LINK_GROUP;
        const int group = threadIdx.x / LINK_GROUP;
        const int member = threadIdx.x % LINK_GROUP;
        for (int first_link = 0; first_link < NumLinks; first_link += GroupsPerBlock) {
          const int link_idx = first_link + group;
          float v[CONTRIB_STRIDE] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
          if (link_idx < NumLinks) {
            if constexpr (NumSpheres > 0) {
              const int first = sorted_map ? link_sphere_range[2 * link_idx] : 0;
              const int last = sorted_map ? link_sphere_range[2 * link_idx + 1] : NumSpheres;
              for (int sphere_idx = first + member; sphere_idx < last;
                   sphere_idx += LINK_GROUP) {
                if (!sorted_map && link_sphere_map[sphere_idx] != link_idx) {
                  continue;
                }
                const float *slot = &contrib[sphere_idx * CONTRIB_STRIDE];
                #pragma unroll
                for (int k = 0; k < CONTRIB_STRIDE; ++k) {
                  v[k] += slot[k];
                }
              }
            }
            for (int tool_idx = member; tool_idx < NumToolFrames; tool_idx += LINK_GROUP) {
              if (tool_frame_map[tool_idx] != link_idx) {
                continue;
              }
              const float *slot = &contrib[(NumSpheres + tool_idx) * CONTRIB_STRIDE];
              #pragma unroll
              for (int k = 0; k < CONTRIB_STRIDE; ++k) {
                v[k] += slot[k];
              }
            }
          }
          #pragma unroll
          for (int k = 0; k < CONTRIB_STRIDE; ++k) {
            #pragma unroll
            for (int offset = LINK_GROUP / 2; offset > 0; offset /= 2) {
              v[k] += __shfl_xor_sync(0xffffffffu, v[k], offset, LINK_GROUP);
            }
          }
          if (link_idx < NumLinks && member == 0) {
            const int wrench_offset = link_idx * WRENCH_STRIDE;
            shared_wrench[wrench_offset + 0] = v[0];
            shared_wrench[wrench_offset + 1] = v[1];
            shared_wrench[wrench_offset + 2] = v[2];
            shared_wrench[wrench_offset + 4] = v[3];
            shared_wrench[wrench_offset + 5] = v[4];
            shared_wrench[wrench_offset + 6] = v[5];
          }
        }
      }

      if constexpr (ComputeCom) {
        __syncthreads();
        const float total_mass = batch_center_of_mass[batch * 4 + 3];
        const float3 com_grad = make_float3(
          grad_center_of_mass[batch * 4 + 0],
          grad_center_of_mass[batch * 4 + 1],
          grad_center_of_mass[batch * 4 + 2]);
        if (total_mass > 0.0f &&
            (com_grad.x != 0.0f || com_grad.y != 0.0f || com_grad.z != 0.0f)) {
          for (int link_idx = threadIdx.x; link_idx < NumLinks;
               link_idx += ThreadsPerBlock) {
            const float4 mass_com = *(const float4 *)&link_masses_com[link_idx * 4];
            if (mass_com.w <= 0.0f) {
              continue;
            }
            const float mass_scale = mass_com.w / total_mass;
            const float3 force = mass_scale * com_grad;
            float4 com_world;
            transform_sphere_float4(
              &global_cumul_mat[batch_matrix_offset + link_idx * MATRIX_STRIDE],
              mass_com,
              com_world);
            const float3 moment = cross(make_float3(com_world), force);
            const int wrench_offset = link_idx * WRENCH_STRIDE;
            shared_wrench[wrench_offset + 0] += moment.x;
            shared_wrench[wrench_offset + 1] += moment.y;
            shared_wrench[wrench_offset + 2] += moment.z;
            shared_wrench[wrench_offset + 4] += force.x;
            shared_wrench[wrench_offset + 5] += force.y;
            shared_wrench[wrench_offset + 6] += force.z;
          }
        }
      }

      __syncthreads();
      if (threadIdx.x >= 32) {
        return;
      }

      const int lane = threadIdx.x;
      for (int joint_idx = lane; joint_idx < NumJoints; joint_idx += 32) {
        float joint_gradient = 0.0f;
        const int joint_link_start = SingleLinkPerJoint
          ? joint_idx
          : joint_links_offsets[joint_idx];
        const int joint_link_end = SingleLinkPerJoint
          ? joint_idx + 1
          : joint_links_offsets[joint_idx + 1];
        for (int joint_link_index = joint_link_start;
             joint_link_index < joint_link_end;
             ++joint_link_index) {
          const int joint_link = joint_links_data[joint_link_index];
          float3 subtree_moment = make_float3(0.0f, 0.0f, 0.0f);
          float3 subtree_force = make_float3(0.0f, 0.0f, 0.0f);
          constexpr int InfluenceWordCount = (NumLinks + 31) / 32;
          const int influence_offset = joint_link * InfluenceWordCount;
          #pragma unroll
          for (int word_index = 0; word_index < InfluenceWordCount; ++word_index) {
            const uint32_t influence_word =
              joint_link_influence_words[influence_offset + word_index];
            #pragma unroll
            for (int bit_index = 0; bit_index < 32; ++bit_index) {
              const int output_link = word_index * 32 + bit_index;
              if (output_link >= NumLinks ||
                  (influence_word & (uint32_t(1) << bit_index)) == 0) {
                continue;
              }
              const int wrench_offset = output_link * WRENCH_STRIDE;
              const float4 moment_data = *(const float4 *)&shared_wrench[wrench_offset];
              const float4 force_data =
                *(const float4 *)&shared_wrench[wrench_offset + 4];
              subtree_moment.x += moment_data.x;
              subtree_moment.y += moment_data.y;
              subtree_moment.z += moment_data.z;
              subtree_force.x += force_data.x;
              subtree_force.y += force_data.y;
              subtree_force.z += force_data.z;
            }
          }

          const float *joint_transform = &cumul_mat[joint_link * MATRIX_STRIDE];
          const int joint_type = joint_map_type[joint_link];
          const int axis_index = joint_type <= Z_PRISM
            ? joint_type
            : joint_type - X_ROT;
          const float3 axis = make_float3(
            joint_transform[axis_index],
            joint_transform[4 + axis_index],
            joint_transform[8 + axis_index]);
          const float axis_scale = joint_offset[joint_link * 2];
          if (joint_type >= X_ROT && joint_type <= Z_ROT) {
            const float3 joint_origin = make_float3(
              joint_transform[3], joint_transform[7], joint_transform[11]);
            joint_gradient += axis_scale * dot(
              axis, subtree_moment - cross(joint_origin, subtree_force));
          } else if (joint_type >= X_PRISM && joint_type <= Z_PRISM) {
            joint_gradient += axis_scale * dot(axis, subtree_force);
          }
        }
        grad_out_link_q[batch * NumJoints + joint_idx] = joint_gradient;
      }
    }

}
}
