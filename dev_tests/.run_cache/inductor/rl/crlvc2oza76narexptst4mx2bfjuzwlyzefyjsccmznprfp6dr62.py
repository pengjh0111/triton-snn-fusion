
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 29, 'num_store': 9, 'num_reduction': 36, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 559104}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr1 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr2 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp40 = tl.load(in_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr1 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp43 = tl.load(in_ptr2 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp59 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp60 = tl.load(in_ptr1 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp62 = tl.load(in_ptr2 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp78 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp79 = tl.load(in_ptr1 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp81 = tl.load(in_ptr2 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp97 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp98 = tl.load(in_ptr1 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp100 = tl.load(in_ptr2 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp116 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp117 = tl.load(in_ptr1 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp119 = tl.load(in_ptr2 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp135 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp136 = tl.load(in_ptr1 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp138 = tl.load(in_ptr2 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp154 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp155 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp157 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp180 = tl.load(in_ptr11 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp182 = tl.load(in_ptr12 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(r0_mask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
    tmp33 = (tmp32 / tmp13)
    tmp34 = tmp26 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
    tmp38 = tl.where(r0_mask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None].to(tl.float32)
    tmp42 = tmp40 + tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask & xmask, tmp45, 0)
    tmp48 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask & xmask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp52 = (tmp51 / tmp13)
    tmp53 = tmp45 - tmp52
    tmp54 = tmp53 * tmp53
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask & xmask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp66 = tl.where(r0_mask & xmask, tmp64, 0)
    tmp67 = tl.broadcast_to(tmp64, [XBLOCK, R0_BLOCK])
    tmp69 = tl.where(r0_mask & xmask, tmp67, 0)
    tmp70 = tl.sum(tmp69, 1)[:, None].to(tl.float32)
    tmp71 = (tmp70 / tmp13)
    tmp72 = tmp64 - tmp71
    tmp73 = tmp72 * tmp72
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
    tmp76 = tl.where(r0_mask & xmask, tmp74, 0)
    tmp77 = tl.sum(tmp76, 1)[:, None].to(tl.float32)
    tmp80 = tmp78 + tmp79
    tmp82 = tmp80 + tmp81
    tmp83 = tl.broadcast_to(tmp82, [XBLOCK, R0_BLOCK])
    tmp85 = tl.where(r0_mask & xmask, tmp83, 0)
    tmp86 = tl.broadcast_to(tmp83, [XBLOCK, R0_BLOCK])
    tmp88 = tl.where(r0_mask & xmask, tmp86, 0)
    tmp89 = tl.sum(tmp88, 1)[:, None].to(tl.float32)
    tmp90 = (tmp89 / tmp13)
    tmp91 = tmp83 - tmp90
    tmp92 = tmp91 * tmp91
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, R0_BLOCK])
    tmp95 = tl.where(r0_mask & xmask, tmp93, 0)
    tmp96 = tl.sum(tmp95, 1)[:, None].to(tl.float32)
    tmp99 = tmp97 + tmp98
    tmp101 = tmp99 + tmp100
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK, R0_BLOCK])
    tmp104 = tl.where(r0_mask & xmask, tmp102, 0)
    tmp105 = tl.broadcast_to(tmp102, [XBLOCK, R0_BLOCK])
    tmp107 = tl.where(r0_mask & xmask, tmp105, 0)
    tmp108 = tl.sum(tmp107, 1)[:, None].to(tl.float32)
    tmp109 = (tmp108 / tmp13)
    tmp110 = tmp102 - tmp109
    tmp111 = tmp110 * tmp110
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK, R0_BLOCK])
    tmp114 = tl.where(r0_mask & xmask, tmp112, 0)
    tmp115 = tl.sum(tmp114, 1)[:, None].to(tl.float32)
    tmp118 = tmp116 + tmp117
    tmp120 = tmp118 + tmp119
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK, R0_BLOCK])
    tmp123 = tl.where(r0_mask & xmask, tmp121, 0)
    tmp124 = tl.broadcast_to(tmp121, [XBLOCK, R0_BLOCK])
    tmp126 = tl.where(r0_mask & xmask, tmp124, 0)
    tmp127 = tl.sum(tmp126, 1)[:, None].to(tl.float32)
    tmp128 = (tmp127 / tmp13)
    tmp129 = tmp121 - tmp128
    tmp130 = tmp129 * tmp129
    tmp131 = tl.broadcast_to(tmp130, [XBLOCK, R0_BLOCK])
    tmp133 = tl.where(r0_mask & xmask, tmp131, 0)
    tmp134 = tl.sum(tmp133, 1)[:, None].to(tl.float32)
    tmp137 = tmp135 + tmp136
    tmp139 = tmp137 + tmp138
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK, R0_BLOCK])
    tmp142 = tl.where(r0_mask & xmask, tmp140, 0)
    tmp143 = tl.broadcast_to(tmp140, [XBLOCK, R0_BLOCK])
    tmp145 = tl.where(r0_mask & xmask, tmp143, 0)
    tmp146 = tl.sum(tmp145, 1)[:, None].to(tl.float32)
    tmp147 = (tmp146 / tmp13)
    tmp148 = tmp140 - tmp147
    tmp149 = tmp148 * tmp148
    tmp150 = tl.broadcast_to(tmp149, [XBLOCK, R0_BLOCK])
    tmp152 = tl.where(r0_mask & xmask, tmp150, 0)
    tmp153 = tl.sum(tmp152, 1)[:, None].to(tl.float32)
    tmp156 = tmp154 + tmp155
    tmp158 = tmp156 + tmp157
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, R0_BLOCK])
    tmp161 = tl.where(r0_mask & xmask, tmp159, 0)
    tmp162 = tl.broadcast_to(tmp159, [XBLOCK, R0_BLOCK])
    tmp164 = tl.where(r0_mask & xmask, tmp162, 0)
    tmp165 = tl.sum(tmp164, 1)[:, None].to(tl.float32)
    tmp166 = (tmp165 / tmp13)
    tmp167 = tmp159 - tmp166
    tmp168 = tmp167 * tmp167
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, R0_BLOCK])
    tmp171 = tl.where(r0_mask & xmask, tmp169, 0)
    tmp172 = tl.sum(tmp171, 1)[:, None].to(tl.float32)
    tmp173 = tmp158 - tmp166
    tmp174 = tl.full([1, 1], 768.0, tl.float32)
    tmp175 = (tmp172 / tmp174)
    tmp176 = tl.full([1, 1], 1e-05, tl.float32)
    tmp177 = tmp175 + tmp176
    tmp178 = libdevice.rsqrt(tmp177)
    tmp179 = tmp173 * tmp178
    tmp181 = tmp179 * tmp180
    tmp183 = tmp181 + tmp182
    tmp184 = tmp139 - tmp147
    tmp185 = (tmp153 / tmp174)
    tmp186 = tmp185 + tmp176
    tmp187 = libdevice.rsqrt(tmp186)
    tmp188 = tmp184 * tmp187
    tmp189 = tmp188 * tmp180
    tmp190 = tmp189 + tmp182
    tmp191 = tmp120 - tmp128
    tmp192 = (tmp134 / tmp174)
    tmp193 = tmp192 + tmp176
    tmp194 = libdevice.rsqrt(tmp193)
    tmp195 = tmp191 * tmp194
    tmp196 = tmp195 * tmp180
    tmp197 = tmp196 + tmp182
    tmp198 = tmp101 - tmp109
    tmp199 = (tmp115 / tmp174)
    tmp200 = tmp199 + tmp176
    tmp201 = libdevice.rsqrt(tmp200)
    tmp202 = tmp198 * tmp201
    tmp203 = tmp202 * tmp180
    tmp204 = tmp203 + tmp182
    tmp205 = tmp82 - tmp90
    tmp206 = (tmp96 / tmp174)
    tmp207 = tmp206 + tmp176
    tmp208 = libdevice.rsqrt(tmp207)
    tmp209 = tmp205 * tmp208
    tmp210 = tmp209 * tmp180
    tmp211 = tmp210 + tmp182
    tmp212 = tmp63 - tmp71
    tmp213 = (tmp77 / tmp174)
    tmp214 = tmp213 + tmp176
    tmp215 = libdevice.rsqrt(tmp214)
    tmp216 = tmp212 * tmp215
    tmp217 = tmp216 * tmp180
    tmp218 = tmp217 + tmp182
    tmp219 = tmp44 - tmp52
    tmp220 = (tmp58 / tmp174)
    tmp221 = tmp220 + tmp176
    tmp222 = libdevice.rsqrt(tmp221)
    tmp223 = tmp219 * tmp222
    tmp224 = tmp223 * tmp180
    tmp225 = tmp224 + tmp182
    tmp226 = tmp25 - tmp33
    tmp227 = (tmp39 / tmp174)
    tmp228 = tmp227 + tmp176
    tmp229 = libdevice.rsqrt(tmp228)
    tmp230 = tmp226 * tmp229
    tmp231 = tmp230 * tmp180
    tmp232 = tmp231 + tmp182
    tmp233 = tmp4 - tmp14
    tmp234 = (tmp20 / tmp174)
    tmp235 = tmp234 + tmp176
    tmp236 = libdevice.rsqrt(tmp235)
    tmp237 = tmp233 * tmp236
    tmp238 = tmp237 * tmp180
    tmp239 = tmp238 + tmp182
    tl.store(out_ptr18 + (r0_1 + 768*x0), tmp183, r0_mask & xmask)
    tl.store(out_ptr19 + (r0_1 + 768*x0), tmp190, r0_mask & xmask)
    tl.store(out_ptr20 + (r0_1 + 768*x0), tmp197, r0_mask & xmask)
    tl.store(out_ptr21 + (r0_1 + 768*x0), tmp204, r0_mask & xmask)
    tl.store(out_ptr22 + (r0_1 + 768*x0), tmp211, r0_mask & xmask)
    tl.store(out_ptr23 + (r0_1 + 768*x0), tmp218, r0_mask & xmask)
    tl.store(out_ptr24 + (r0_1 + 768*x0), tmp225, r0_mask & xmask)
    tl.store(out_ptr25 + (r0_1 + 768*x0), tmp232, r0_mask & xmask)
    tl.store(out_ptr26 + (r0_1 + 768*x0), tmp239, r0_mask & xmask)
