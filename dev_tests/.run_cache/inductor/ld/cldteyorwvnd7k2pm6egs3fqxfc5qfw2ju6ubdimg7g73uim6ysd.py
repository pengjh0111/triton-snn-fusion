
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 30, 'num_store': 7, 'num_reduction': 28, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 522240}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr0 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp44 = tl.load(in_ptr0 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr1 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp47 = tl.load(in_ptr2 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp49 = tl.load(in_ptr3 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp65 = tl.load(in_ptr0 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp66 = tl.load(in_ptr1 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr2 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp70 = tl.load(in_ptr3 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp86 = tl.load(in_ptr0 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr1 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp89 = tl.load(in_ptr2 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp91 = tl.load(in_ptr3 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp107 = tl.load(in_ptr0 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp108 = tl.load(in_ptr1 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp110 = tl.load(in_ptr2 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp112 = tl.load(in_ptr3 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp128 = tl.load(in_ptr0 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp129 = tl.load(in_ptr1 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp131 = tl.load(in_ptr2 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp133 = tl.load(in_ptr3 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp156 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp158 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(r0_mask & xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp35 = tl.where(r0_mask & xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None].to(tl.float32)
    tmp37 = (tmp36 / tmp15)
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
    tmp42 = tl.where(r0_mask & xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None].to(tl.float32)
    tmp46 = tmp44 + tmp45
    tmp48 = tmp46 + tmp47
    tmp50 = tmp48 + tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(r0_mask & xmask, tmp51, 0)
    tmp54 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp56 = tl.where(r0_mask & xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None].to(tl.float32)
    tmp58 = (tmp57 / tmp15)
    tmp59 = tmp51 - tmp58
    tmp60 = tmp59 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, R0_BLOCK])
    tmp63 = tl.where(r0_mask & xmask, tmp61, 0)
    tmp64 = tl.sum(tmp63, 1)[:, None].to(tl.float32)
    tmp67 = tmp65 + tmp66
    tmp69 = tmp67 + tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, R0_BLOCK])
    tmp74 = tl.where(r0_mask & xmask, tmp72, 0)
    tmp75 = tl.broadcast_to(tmp72, [XBLOCK, R0_BLOCK])
    tmp77 = tl.where(r0_mask & xmask, tmp75, 0)
    tmp78 = tl.sum(tmp77, 1)[:, None].to(tl.float32)
    tmp79 = (tmp78 / tmp15)
    tmp80 = tmp72 - tmp79
    tmp81 = tmp80 * tmp80
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, R0_BLOCK])
    tmp84 = tl.where(r0_mask & xmask, tmp82, 0)
    tmp85 = tl.sum(tmp84, 1)[:, None].to(tl.float32)
    tmp88 = tmp86 + tmp87
    tmp90 = tmp88 + tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, R0_BLOCK])
    tmp95 = tl.where(r0_mask & xmask, tmp93, 0)
    tmp96 = tl.broadcast_to(tmp93, [XBLOCK, R0_BLOCK])
    tmp98 = tl.where(r0_mask & xmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None].to(tl.float32)
    tmp100 = (tmp99 / tmp15)
    tmp101 = tmp93 - tmp100
    tmp102 = tmp101 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, R0_BLOCK])
    tmp105 = tl.where(r0_mask & xmask, tmp103, 0)
    tmp106 = tl.sum(tmp105, 1)[:, None].to(tl.float32)
    tmp109 = tmp107 + tmp108
    tmp111 = tmp109 + tmp110
    tmp113 = tmp111 + tmp112
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK, R0_BLOCK])
    tmp116 = tl.where(r0_mask & xmask, tmp114, 0)
    tmp117 = tl.broadcast_to(tmp114, [XBLOCK, R0_BLOCK])
    tmp119 = tl.where(r0_mask & xmask, tmp117, 0)
    tmp120 = tl.sum(tmp119, 1)[:, None].to(tl.float32)
    tmp121 = (tmp120 / tmp15)
    tmp122 = tmp114 - tmp121
    tmp123 = tmp122 * tmp122
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, R0_BLOCK])
    tmp126 = tl.where(r0_mask & xmask, tmp124, 0)
    tmp127 = tl.sum(tmp126, 1)[:, None].to(tl.float32)
    tmp130 = tmp128 + tmp129
    tmp132 = tmp130 + tmp131
    tmp134 = tmp132 + tmp133
    tmp135 = tl.broadcast_to(tmp134, [XBLOCK, R0_BLOCK])
    tmp137 = tl.where(r0_mask & xmask, tmp135, 0)
    tmp138 = tl.broadcast_to(tmp135, [XBLOCK, R0_BLOCK])
    tmp140 = tl.where(r0_mask & xmask, tmp138, 0)
    tmp141 = tl.sum(tmp140, 1)[:, None].to(tl.float32)
    tmp142 = (tmp141 / tmp15)
    tmp143 = tmp135 - tmp142
    tmp144 = tmp143 * tmp143
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK, R0_BLOCK])
    tmp147 = tl.where(r0_mask & xmask, tmp145, 0)
    tmp148 = tl.sum(tmp147, 1)[:, None].to(tl.float32)
    tmp149 = tmp134 - tmp142
    tmp150 = tl.full([1, 1], 768.0, tl.float32)
    tmp151 = (tmp148 / tmp150)
    tmp152 = tl.full([1, 1], 1e-05, tl.float32)
    tmp153 = tmp151 + tmp152
    tmp154 = libdevice.rsqrt(tmp153)
    tmp155 = tmp149 * tmp154
    tmp157 = tmp155 * tmp156
    tmp159 = tmp157 + tmp158
    tmp160 = tmp113 - tmp121
    tmp161 = (tmp127 / tmp150)
    tmp162 = tmp161 + tmp152
    tmp163 = libdevice.rsqrt(tmp162)
    tmp164 = tmp160 * tmp163
    tmp165 = tmp164 * tmp156
    tmp166 = tmp165 + tmp158
    tmp167 = tmp92 - tmp100
    tmp168 = (tmp106 / tmp150)
    tmp169 = tmp168 + tmp152
    tmp170 = libdevice.rsqrt(tmp169)
    tmp171 = tmp167 * tmp170
    tmp172 = tmp171 * tmp156
    tmp173 = tmp172 + tmp158
    tmp174 = tmp71 - tmp79
    tmp175 = (tmp85 / tmp150)
    tmp176 = tmp175 + tmp152
    tmp177 = libdevice.rsqrt(tmp176)
    tmp178 = tmp174 * tmp177
    tmp179 = tmp178 * tmp156
    tmp180 = tmp179 + tmp158
    tmp181 = tmp50 - tmp58
    tmp182 = (tmp64 / tmp150)
    tmp183 = tmp182 + tmp152
    tmp184 = libdevice.rsqrt(tmp183)
    tmp185 = tmp181 * tmp184
    tmp186 = tmp185 * tmp156
    tmp187 = tmp186 + tmp158
    tmp188 = tmp29 - tmp37
    tmp189 = (tmp43 / tmp150)
    tmp190 = tmp189 + tmp152
    tmp191 = libdevice.rsqrt(tmp190)
    tmp192 = tmp188 * tmp191
    tmp193 = tmp192 * tmp156
    tmp194 = tmp193 + tmp158
    tmp195 = tmp6 - tmp16
    tmp196 = (tmp22 / tmp150)
    tmp197 = tmp196 + tmp152
    tmp198 = libdevice.rsqrt(tmp197)
    tmp199 = tmp195 * tmp198
    tmp200 = tmp199 * tmp156
    tmp201 = tmp200 + tmp158
    tl.store(out_ptr14 + (r0_1 + 768*x0), tmp159, r0_mask & xmask)
    tl.store(out_ptr15 + (r0_1 + 768*x0), tmp166, r0_mask & xmask)
    tl.store(out_ptr16 + (r0_1 + 768*x0), tmp173, r0_mask & xmask)
    tl.store(out_ptr17 + (r0_1 + 768*x0), tmp180, r0_mask & xmask)
    tl.store(out_ptr18 + (r0_1 + 768*x0), tmp187, r0_mask & xmask)
    tl.store(out_ptr19 + (r0_1 + 768*x0), tmp194, r0_mask & xmask)
    tl.store(out_ptr20 + (r0_1 + 768*x0), tmp201, r0_mask & xmask)
