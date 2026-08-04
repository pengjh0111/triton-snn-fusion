
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 20, 'num_store': 9, 'num_reduction': 36, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 448512}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr1 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr0 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr1 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp53 = tl.load(in_ptr0 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp54 = tl.load(in_ptr1 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp70 = tl.load(in_ptr0 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr0 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp88 = tl.load(in_ptr1 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp104 = tl.load(in_ptr0 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp105 = tl.load(in_ptr1 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp121 = tl.load(in_ptr0 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp122 = tl.load(in_ptr1 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp138 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp139 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp162 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp164 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = (tmp28 / tmp11)
    tmp30 = tmp22 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp38 = tmp36 + tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
    tmp44 = tl.where(r0_mask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None].to(tl.float32)
    tmp46 = (tmp45 / tmp11)
    tmp47 = tmp39 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, R0_BLOCK])
    tmp51 = tl.where(r0_mask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tmp55 = tmp53 + tmp54
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
    tmp58 = tl.where(r0_mask & xmask, tmp56, 0)
    tmp59 = tl.broadcast_to(tmp56, [XBLOCK, R0_BLOCK])
    tmp61 = tl.where(r0_mask & xmask, tmp59, 0)
    tmp62 = tl.sum(tmp61, 1)[:, None].to(tl.float32)
    tmp63 = (tmp62 / tmp11)
    tmp64 = tmp56 - tmp63
    tmp65 = tmp64 * tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(r0_mask & xmask, tmp66, 0)
    tmp69 = tl.sum(tmp68, 1)[:, None].to(tl.float32)
    tmp72 = tmp70 + tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, R0_BLOCK])
    tmp75 = tl.where(r0_mask & xmask, tmp73, 0)
    tmp76 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
    tmp78 = tl.where(r0_mask & xmask, tmp76, 0)
    tmp79 = tl.sum(tmp78, 1)[:, None].to(tl.float32)
    tmp80 = (tmp79 / tmp11)
    tmp81 = tmp73 - tmp80
    tmp82 = tmp81 * tmp81
    tmp83 = tl.broadcast_to(tmp82, [XBLOCK, R0_BLOCK])
    tmp85 = tl.where(r0_mask & xmask, tmp83, 0)
    tmp86 = tl.sum(tmp85, 1)[:, None].to(tl.float32)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
    tmp92 = tl.where(r0_mask & xmask, tmp90, 0)
    tmp93 = tl.broadcast_to(tmp90, [XBLOCK, R0_BLOCK])
    tmp95 = tl.where(r0_mask & xmask, tmp93, 0)
    tmp96 = tl.sum(tmp95, 1)[:, None].to(tl.float32)
    tmp97 = (tmp96 / tmp11)
    tmp98 = tmp90 - tmp97
    tmp99 = tmp98 * tmp98
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, R0_BLOCK])
    tmp102 = tl.where(r0_mask & xmask, tmp100, 0)
    tmp103 = tl.sum(tmp102, 1)[:, None].to(tl.float32)
    tmp106 = tmp104 + tmp105
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, R0_BLOCK])
    tmp109 = tl.where(r0_mask & xmask, tmp107, 0)
    tmp110 = tl.broadcast_to(tmp107, [XBLOCK, R0_BLOCK])
    tmp112 = tl.where(r0_mask & xmask, tmp110, 0)
    tmp113 = tl.sum(tmp112, 1)[:, None].to(tl.float32)
    tmp114 = (tmp113 / tmp11)
    tmp115 = tmp107 - tmp114
    tmp116 = tmp115 * tmp115
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK, R0_BLOCK])
    tmp119 = tl.where(r0_mask & xmask, tmp117, 0)
    tmp120 = tl.sum(tmp119, 1)[:, None].to(tl.float32)
    tmp123 = tmp121 + tmp122
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, R0_BLOCK])
    tmp126 = tl.where(r0_mask & xmask, tmp124, 0)
    tmp127 = tl.broadcast_to(tmp124, [XBLOCK, R0_BLOCK])
    tmp129 = tl.where(r0_mask & xmask, tmp127, 0)
    tmp130 = tl.sum(tmp129, 1)[:, None].to(tl.float32)
    tmp131 = (tmp130 / tmp11)
    tmp132 = tmp124 - tmp131
    tmp133 = tmp132 * tmp132
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK, R0_BLOCK])
    tmp136 = tl.where(r0_mask & xmask, tmp134, 0)
    tmp137 = tl.sum(tmp136, 1)[:, None].to(tl.float32)
    tmp140 = tmp138 + tmp139
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK, R0_BLOCK])
    tmp143 = tl.where(r0_mask & xmask, tmp141, 0)
    tmp144 = tl.broadcast_to(tmp141, [XBLOCK, R0_BLOCK])
    tmp146 = tl.where(r0_mask & xmask, tmp144, 0)
    tmp147 = tl.sum(tmp146, 1)[:, None].to(tl.float32)
    tmp148 = (tmp147 / tmp11)
    tmp149 = tmp141 - tmp148
    tmp150 = tmp149 * tmp149
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, R0_BLOCK])
    tmp153 = tl.where(r0_mask & xmask, tmp151, 0)
    tmp154 = tl.sum(tmp153, 1)[:, None].to(tl.float32)
    tmp155 = tmp140 - tmp148
    tmp156 = tl.full([1, 1], 768.0, tl.float32)
    tmp157 = (tmp154 / tmp156)
    tmp158 = tl.full([1, 1], 1e-05, tl.float32)
    tmp159 = tmp157 + tmp158
    tmp160 = libdevice.rsqrt(tmp159)
    tmp161 = tmp155 * tmp160
    tmp163 = tmp161 * tmp162
    tmp165 = tmp163 + tmp164
    tmp166 = tmp123 - tmp131
    tmp167 = (tmp137 / tmp156)
    tmp168 = tmp167 + tmp158
    tmp169 = libdevice.rsqrt(tmp168)
    tmp170 = tmp166 * tmp169
    tmp171 = tmp170 * tmp162
    tmp172 = tmp171 + tmp164
    tmp173 = tmp106 - tmp114
    tmp174 = (tmp120 / tmp156)
    tmp175 = tmp174 + tmp158
    tmp176 = libdevice.rsqrt(tmp175)
    tmp177 = tmp173 * tmp176
    tmp178 = tmp177 * tmp162
    tmp179 = tmp178 + tmp164
    tmp180 = tmp89 - tmp97
    tmp181 = (tmp103 / tmp156)
    tmp182 = tmp181 + tmp158
    tmp183 = libdevice.rsqrt(tmp182)
    tmp184 = tmp180 * tmp183
    tmp185 = tmp184 * tmp162
    tmp186 = tmp185 + tmp164
    tmp187 = tmp72 - tmp80
    tmp188 = (tmp86 / tmp156)
    tmp189 = tmp188 + tmp158
    tmp190 = libdevice.rsqrt(tmp189)
    tmp191 = tmp187 * tmp190
    tmp192 = tmp191 * tmp162
    tmp193 = tmp192 + tmp164
    tmp194 = tmp55 - tmp63
    tmp195 = (tmp69 / tmp156)
    tmp196 = tmp195 + tmp158
    tmp197 = libdevice.rsqrt(tmp196)
    tmp198 = tmp194 * tmp197
    tmp199 = tmp198 * tmp162
    tmp200 = tmp199 + tmp164
    tmp201 = tmp38 - tmp46
    tmp202 = (tmp52 / tmp156)
    tmp203 = tmp202 + tmp158
    tmp204 = libdevice.rsqrt(tmp203)
    tmp205 = tmp201 * tmp204
    tmp206 = tmp205 * tmp162
    tmp207 = tmp206 + tmp164
    tmp208 = tmp21 - tmp29
    tmp209 = (tmp35 / tmp156)
    tmp210 = tmp209 + tmp158
    tmp211 = libdevice.rsqrt(tmp210)
    tmp212 = tmp208 * tmp211
    tmp213 = tmp212 * tmp162
    tmp214 = tmp213 + tmp164
    tmp215 = tmp2 - tmp12
    tmp216 = (tmp18 / tmp156)
    tmp217 = tmp216 + tmp158
    tmp218 = libdevice.rsqrt(tmp217)
    tmp219 = tmp215 * tmp218
    tmp220 = tmp219 * tmp162
    tmp221 = tmp220 + tmp164
    tl.store(out_ptr18 + (r0_1 + 768*x0), tmp165, r0_mask & xmask)
    tl.store(out_ptr19 + (r0_1 + 768*x0), tmp172, r0_mask & xmask)
    tl.store(out_ptr20 + (r0_1 + 768*x0), tmp179, r0_mask & xmask)
    tl.store(out_ptr21 + (r0_1 + 768*x0), tmp186, r0_mask & xmask)
    tl.store(out_ptr22 + (r0_1 + 768*x0), tmp193, r0_mask & xmask)
    tl.store(out_ptr23 + (r0_1 + 768*x0), tmp200, r0_mask & xmask)
    tl.store(out_ptr24 + (r0_1 + 768*x0), tmp207, r0_mask & xmask)
    tl.store(out_ptr25 + (r0_1 + 768*x0), tmp214, r0_mask & xmask)
    tl.store(out_ptr26 + (r0_1 + 768*x0), tmp221, r0_mask & xmask)
