
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 38, 'num_store': 9, 'num_reduction': 36, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 669696}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr3 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp44 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr1 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp47 = tl.load(in_ptr2 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp49 = tl.load(in_ptr3 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp65 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp66 = tl.load(in_ptr1 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr2 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp70 = tl.load(in_ptr3 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp86 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr1 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp89 = tl.load(in_ptr2 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp91 = tl.load(in_ptr3 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp107 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp108 = tl.load(in_ptr1 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp110 = tl.load(in_ptr2 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp112 = tl.load(in_ptr3 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp128 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp129 = tl.load(in_ptr1 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp131 = tl.load(in_ptr2 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp133 = tl.load(in_ptr3 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp149 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp150 = tl.load(in_ptr1 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp152 = tl.load(in_ptr2 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp154 = tl.load(in_ptr3 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp170 = tl.load(in_ptr11 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp171 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp173 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp175 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp198 = tl.load(in_ptr12 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp200 = tl.load(in_ptr13 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tmp151 = tmp149 + tmp150
    tmp153 = tmp151 + tmp152
    tmp155 = tmp153 + tmp154
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK, R0_BLOCK])
    tmp158 = tl.where(r0_mask & xmask, tmp156, 0)
    tmp159 = tl.broadcast_to(tmp156, [XBLOCK, R0_BLOCK])
    tmp161 = tl.where(r0_mask & xmask, tmp159, 0)
    tmp162 = tl.sum(tmp161, 1)[:, None].to(tl.float32)
    tmp163 = (tmp162 / tmp15)
    tmp164 = tmp156 - tmp163
    tmp165 = tmp164 * tmp164
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, R0_BLOCK])
    tmp168 = tl.where(r0_mask & xmask, tmp166, 0)
    tmp169 = tl.sum(tmp168, 1)[:, None].to(tl.float32)
    tmp172 = tmp170 + tmp171
    tmp174 = tmp172 + tmp173
    tmp176 = tmp174 + tmp175
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, R0_BLOCK])
    tmp179 = tl.where(r0_mask & xmask, tmp177, 0)
    tmp180 = tl.broadcast_to(tmp177, [XBLOCK, R0_BLOCK])
    tmp182 = tl.where(r0_mask & xmask, tmp180, 0)
    tmp183 = tl.sum(tmp182, 1)[:, None].to(tl.float32)
    tmp184 = (tmp183 / tmp15)
    tmp185 = tmp177 - tmp184
    tmp186 = tmp185 * tmp185
    tmp187 = tl.broadcast_to(tmp186, [XBLOCK, R0_BLOCK])
    tmp189 = tl.where(r0_mask & xmask, tmp187, 0)
    tmp190 = tl.sum(tmp189, 1)[:, None].to(tl.float32)
    tmp191 = tmp176 - tmp184
    tmp192 = tl.full([1, 1], 768.0, tl.float32)
    tmp193 = (tmp190 / tmp192)
    tmp194 = tl.full([1, 1], 1e-05, tl.float32)
    tmp195 = tmp193 + tmp194
    tmp196 = libdevice.rsqrt(tmp195)
    tmp197 = tmp191 * tmp196
    tmp199 = tmp197 * tmp198
    tmp201 = tmp199 + tmp200
    tmp202 = tmp155 - tmp163
    tmp203 = (tmp169 / tmp192)
    tmp204 = tmp203 + tmp194
    tmp205 = libdevice.rsqrt(tmp204)
    tmp206 = tmp202 * tmp205
    tmp207 = tmp206 * tmp198
    tmp208 = tmp207 + tmp200
    tmp209 = tmp134 - tmp142
    tmp210 = (tmp148 / tmp192)
    tmp211 = tmp210 + tmp194
    tmp212 = libdevice.rsqrt(tmp211)
    tmp213 = tmp209 * tmp212
    tmp214 = tmp213 * tmp198
    tmp215 = tmp214 + tmp200
    tmp216 = tmp113 - tmp121
    tmp217 = (tmp127 / tmp192)
    tmp218 = tmp217 + tmp194
    tmp219 = libdevice.rsqrt(tmp218)
    tmp220 = tmp216 * tmp219
    tmp221 = tmp220 * tmp198
    tmp222 = tmp221 + tmp200
    tmp223 = tmp92 - tmp100
    tmp224 = (tmp106 / tmp192)
    tmp225 = tmp224 + tmp194
    tmp226 = libdevice.rsqrt(tmp225)
    tmp227 = tmp223 * tmp226
    tmp228 = tmp227 * tmp198
    tmp229 = tmp228 + tmp200
    tmp230 = tmp71 - tmp79
    tmp231 = (tmp85 / tmp192)
    tmp232 = tmp231 + tmp194
    tmp233 = libdevice.rsqrt(tmp232)
    tmp234 = tmp230 * tmp233
    tmp235 = tmp234 * tmp198
    tmp236 = tmp235 + tmp200
    tmp237 = tmp50 - tmp58
    tmp238 = (tmp64 / tmp192)
    tmp239 = tmp238 + tmp194
    tmp240 = libdevice.rsqrt(tmp239)
    tmp241 = tmp237 * tmp240
    tmp242 = tmp241 * tmp198
    tmp243 = tmp242 + tmp200
    tmp244 = tmp29 - tmp37
    tmp245 = (tmp43 / tmp192)
    tmp246 = tmp245 + tmp194
    tmp247 = libdevice.rsqrt(tmp246)
    tmp248 = tmp244 * tmp247
    tmp249 = tmp248 * tmp198
    tmp250 = tmp249 + tmp200
    tmp251 = tmp6 - tmp16
    tmp252 = (tmp22 / tmp192)
    tmp253 = tmp252 + tmp194
    tmp254 = libdevice.rsqrt(tmp253)
    tmp255 = tmp251 * tmp254
    tmp256 = tmp255 * tmp198
    tmp257 = tmp256 + tmp200
    tl.store(out_ptr18 + (r0_1 + 768*x0), tmp201, r0_mask & xmask)
    tl.store(out_ptr19 + (r0_1 + 768*x0), tmp208, r0_mask & xmask)
    tl.store(out_ptr20 + (r0_1 + 768*x0), tmp215, r0_mask & xmask)
    tl.store(out_ptr21 + (r0_1 + 768*x0), tmp222, r0_mask & xmask)
    tl.store(out_ptr22 + (r0_1 + 768*x0), tmp229, r0_mask & xmask)
    tl.store(out_ptr23 + (r0_1 + 768*x0), tmp236, r0_mask & xmask)
    tl.store(out_ptr24 + (r0_1 + 768*x0), tmp243, r0_mask & xmask)
    tl.store(out_ptr25 + (r0_1 + 768*x0), tmp250, r0_mask & xmask)
    tl.store(out_ptr26 + (r0_1 + 768*x0), tmp257, r0_mask & xmask)
