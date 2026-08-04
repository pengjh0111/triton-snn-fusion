
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_out_ptr9': '*fp32', 'in_out_ptr10': '*fp32', 'in_out_ptr11': '*fp32', 'in_out_ptr12': '*fp32', 'in_out_ptr13': '*fp32', 'in_out_ptr14': '*fp32', 'in_out_ptr15': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_51', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr10', 'in_out_ptr11', 'in_out_ptr12', 'in_out_ptr13', 'in_out_ptr14', 'in_out_ptr15', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 66, 'num_store': 16, 'num_reduction': 64, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1185792}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_51(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_out_ptr10, in_out_ptr11, in_out_ptr12, in_out_ptr13, in_out_ptr14, in_out_ptr15, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr15 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_out_ptr14 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr0 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp44 = tl.load(in_out_ptr13 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr0 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp47 = tl.load(in_ptr1 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp49 = tl.load(in_ptr2 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp65 = tl.load(in_out_ptr12 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp66 = tl.load(in_ptr0 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr1 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp70 = tl.load(in_ptr2 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp86 = tl.load(in_out_ptr11 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr0 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp89 = tl.load(in_ptr1 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp91 = tl.load(in_ptr2 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp107 = tl.load(in_out_ptr10 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp108 = tl.load(in_ptr0 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp110 = tl.load(in_ptr1 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp112 = tl.load(in_ptr2 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp128 = tl.load(in_out_ptr9 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp129 = tl.load(in_ptr0 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp131 = tl.load(in_ptr1 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp133 = tl.load(in_ptr2 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp149 = tl.load(in_out_ptr8 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp150 = tl.load(in_ptr0 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp152 = tl.load(in_ptr1 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp154 = tl.load(in_ptr2 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp170 = tl.load(in_out_ptr7 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp171 = tl.load(in_ptr0 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp173 = tl.load(in_ptr1 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp175 = tl.load(in_ptr2 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp191 = tl.load(in_out_ptr6 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp192 = tl.load(in_ptr0 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp194 = tl.load(in_ptr1 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp196 = tl.load(in_ptr2 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp212 = tl.load(in_out_ptr5 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp213 = tl.load(in_ptr0 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp215 = tl.load(in_ptr1 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp217 = tl.load(in_ptr2 + (15360 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp233 = tl.load(in_out_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp234 = tl.load(in_ptr0 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp236 = tl.load(in_ptr1 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp238 = tl.load(in_ptr2 + (12288 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp254 = tl.load(in_out_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp255 = tl.load(in_ptr0 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp257 = tl.load(in_ptr1 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp259 = tl.load(in_ptr2 + (9216 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp275 = tl.load(in_out_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp276 = tl.load(in_ptr0 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp278 = tl.load(in_ptr1 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp280 = tl.load(in_ptr2 + (6144 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp296 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp297 = tl.load(in_ptr0 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp299 = tl.load(in_ptr1 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp301 = tl.load(in_ptr2 + (3072 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp317 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp318 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp320 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp322 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp345 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp347 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tmp193 = tmp191 + tmp192
    tmp195 = tmp193 + tmp194
    tmp197 = tmp195 + tmp196
    tmp198 = tl.broadcast_to(tmp197, [XBLOCK, R0_BLOCK])
    tmp200 = tl.where(r0_mask & xmask, tmp198, 0)
    tmp201 = tl.broadcast_to(tmp198, [XBLOCK, R0_BLOCK])
    tmp203 = tl.where(r0_mask & xmask, tmp201, 0)
    tmp204 = tl.sum(tmp203, 1)[:, None].to(tl.float32)
    tmp205 = (tmp204 / tmp15)
    tmp206 = tmp198 - tmp205
    tmp207 = tmp206 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, R0_BLOCK])
    tmp210 = tl.where(r0_mask & xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None].to(tl.float32)
    tmp214 = tmp212 + tmp213
    tmp216 = tmp214 + tmp215
    tmp218 = tmp216 + tmp217
    tmp219 = tl.broadcast_to(tmp218, [XBLOCK, R0_BLOCK])
    tmp221 = tl.where(r0_mask & xmask, tmp219, 0)
    tmp222 = tl.broadcast_to(tmp219, [XBLOCK, R0_BLOCK])
    tmp224 = tl.where(r0_mask & xmask, tmp222, 0)
    tmp225 = tl.sum(tmp224, 1)[:, None].to(tl.float32)
    tmp226 = (tmp225 / tmp15)
    tmp227 = tmp219 - tmp226
    tmp228 = tmp227 * tmp227
    tmp229 = tl.broadcast_to(tmp228, [XBLOCK, R0_BLOCK])
    tmp231 = tl.where(r0_mask & xmask, tmp229, 0)
    tmp232 = tl.sum(tmp231, 1)[:, None].to(tl.float32)
    tmp235 = tmp233 + tmp234
    tmp237 = tmp235 + tmp236
    tmp239 = tmp237 + tmp238
    tmp240 = tl.broadcast_to(tmp239, [XBLOCK, R0_BLOCK])
    tmp242 = tl.where(r0_mask & xmask, tmp240, 0)
    tmp243 = tl.broadcast_to(tmp240, [XBLOCK, R0_BLOCK])
    tmp245 = tl.where(r0_mask & xmask, tmp243, 0)
    tmp246 = tl.sum(tmp245, 1)[:, None].to(tl.float32)
    tmp247 = (tmp246 / tmp15)
    tmp248 = tmp240 - tmp247
    tmp249 = tmp248 * tmp248
    tmp250 = tl.broadcast_to(tmp249, [XBLOCK, R0_BLOCK])
    tmp252 = tl.where(r0_mask & xmask, tmp250, 0)
    tmp253 = tl.sum(tmp252, 1)[:, None].to(tl.float32)
    tmp256 = tmp254 + tmp255
    tmp258 = tmp256 + tmp257
    tmp260 = tmp258 + tmp259
    tmp261 = tl.broadcast_to(tmp260, [XBLOCK, R0_BLOCK])
    tmp263 = tl.where(r0_mask & xmask, tmp261, 0)
    tmp264 = tl.broadcast_to(tmp261, [XBLOCK, R0_BLOCK])
    tmp266 = tl.where(r0_mask & xmask, tmp264, 0)
    tmp267 = tl.sum(tmp266, 1)[:, None].to(tl.float32)
    tmp268 = (tmp267 / tmp15)
    tmp269 = tmp261 - tmp268
    tmp270 = tmp269 * tmp269
    tmp271 = tl.broadcast_to(tmp270, [XBLOCK, R0_BLOCK])
    tmp273 = tl.where(r0_mask & xmask, tmp271, 0)
    tmp274 = tl.sum(tmp273, 1)[:, None].to(tl.float32)
    tmp277 = tmp275 + tmp276
    tmp279 = tmp277 + tmp278
    tmp281 = tmp279 + tmp280
    tmp282 = tl.broadcast_to(tmp281, [XBLOCK, R0_BLOCK])
    tmp284 = tl.where(r0_mask & xmask, tmp282, 0)
    tmp285 = tl.broadcast_to(tmp282, [XBLOCK, R0_BLOCK])
    tmp287 = tl.where(r0_mask & xmask, tmp285, 0)
    tmp288 = tl.sum(tmp287, 1)[:, None].to(tl.float32)
    tmp289 = (tmp288 / tmp15)
    tmp290 = tmp282 - tmp289
    tmp291 = tmp290 * tmp290
    tmp292 = tl.broadcast_to(tmp291, [XBLOCK, R0_BLOCK])
    tmp294 = tl.where(r0_mask & xmask, tmp292, 0)
    tmp295 = tl.sum(tmp294, 1)[:, None].to(tl.float32)
    tmp298 = tmp296 + tmp297
    tmp300 = tmp298 + tmp299
    tmp302 = tmp300 + tmp301
    tmp303 = tl.broadcast_to(tmp302, [XBLOCK, R0_BLOCK])
    tmp305 = tl.where(r0_mask & xmask, tmp303, 0)
    tmp306 = tl.broadcast_to(tmp303, [XBLOCK, R0_BLOCK])
    tmp308 = tl.where(r0_mask & xmask, tmp306, 0)
    tmp309 = tl.sum(tmp308, 1)[:, None].to(tl.float32)
    tmp310 = (tmp309 / tmp15)
    tmp311 = tmp303 - tmp310
    tmp312 = tmp311 * tmp311
    tmp313 = tl.broadcast_to(tmp312, [XBLOCK, R0_BLOCK])
    tmp315 = tl.where(r0_mask & xmask, tmp313, 0)
    tmp316 = tl.sum(tmp315, 1)[:, None].to(tl.float32)
    tmp319 = tmp317 + tmp318
    tmp321 = tmp319 + tmp320
    tmp323 = tmp321 + tmp322
    tmp324 = tl.broadcast_to(tmp323, [XBLOCK, R0_BLOCK])
    tmp326 = tl.where(r0_mask & xmask, tmp324, 0)
    tmp327 = tl.broadcast_to(tmp324, [XBLOCK, R0_BLOCK])
    tmp329 = tl.where(r0_mask & xmask, tmp327, 0)
    tmp330 = tl.sum(tmp329, 1)[:, None].to(tl.float32)
    tmp331 = (tmp330 / tmp15)
    tmp332 = tmp324 - tmp331
    tmp333 = tmp332 * tmp332
    tmp334 = tl.broadcast_to(tmp333, [XBLOCK, R0_BLOCK])
    tmp336 = tl.where(r0_mask & xmask, tmp334, 0)
    tmp337 = tl.sum(tmp336, 1)[:, None].to(tl.float32)
    tmp338 = tmp323 - tmp331
    tmp339 = tl.full([1, 1], 768.0, tl.float32)
    tmp340 = (tmp337 / tmp339)
    tmp341 = tl.full([1, 1], 1e-05, tl.float32)
    tmp342 = tmp340 + tmp341
    tmp343 = libdevice.rsqrt(tmp342)
    tmp344 = tmp338 * tmp343
    tmp346 = tmp344 * tmp345
    tmp348 = tmp346 + tmp347
    tmp349 = tmp302 - tmp310
    tmp350 = (tmp316 / tmp339)
    tmp351 = tmp350 + tmp341
    tmp352 = libdevice.rsqrt(tmp351)
    tmp353 = tmp349 * tmp352
    tmp354 = tmp353 * tmp345
    tmp355 = tmp354 + tmp347
    tmp356 = tmp281 - tmp289
    tmp357 = (tmp295 / tmp339)
    tmp358 = tmp357 + tmp341
    tmp359 = libdevice.rsqrt(tmp358)
    tmp360 = tmp356 * tmp359
    tmp361 = tmp360 * tmp345
    tmp362 = tmp361 + tmp347
    tmp363 = tmp260 - tmp268
    tmp364 = (tmp274 / tmp339)
    tmp365 = tmp364 + tmp341
    tmp366 = libdevice.rsqrt(tmp365)
    tmp367 = tmp363 * tmp366
    tmp368 = tmp367 * tmp345
    tmp369 = tmp368 + tmp347
    tmp370 = tmp239 - tmp247
    tmp371 = (tmp253 / tmp339)
    tmp372 = tmp371 + tmp341
    tmp373 = libdevice.rsqrt(tmp372)
    tmp374 = tmp370 * tmp373
    tmp375 = tmp374 * tmp345
    tmp376 = tmp375 + tmp347
    tmp377 = tmp218 - tmp226
    tmp378 = (tmp232 / tmp339)
    tmp379 = tmp378 + tmp341
    tmp380 = libdevice.rsqrt(tmp379)
    tmp381 = tmp377 * tmp380
    tmp382 = tmp381 * tmp345
    tmp383 = tmp382 + tmp347
    tmp384 = tmp197 - tmp205
    tmp385 = (tmp211 / tmp339)
    tmp386 = tmp385 + tmp341
    tmp387 = libdevice.rsqrt(tmp386)
    tmp388 = tmp384 * tmp387
    tmp389 = tmp388 * tmp345
    tmp390 = tmp389 + tmp347
    tmp391 = tmp176 - tmp184
    tmp392 = (tmp190 / tmp339)
    tmp393 = tmp392 + tmp341
    tmp394 = libdevice.rsqrt(tmp393)
    tmp395 = tmp391 * tmp394
    tmp396 = tmp395 * tmp345
    tmp397 = tmp396 + tmp347
    tmp398 = tmp155 - tmp163
    tmp399 = (tmp169 / tmp339)
    tmp400 = tmp399 + tmp341
    tmp401 = libdevice.rsqrt(tmp400)
    tmp402 = tmp398 * tmp401
    tmp403 = tmp402 * tmp345
    tmp404 = tmp403 + tmp347
    tmp405 = tmp134 - tmp142
    tmp406 = (tmp148 / tmp339)
    tmp407 = tmp406 + tmp341
    tmp408 = libdevice.rsqrt(tmp407)
    tmp409 = tmp405 * tmp408
    tmp410 = tmp409 * tmp345
    tmp411 = tmp410 + tmp347
    tmp412 = tmp113 - tmp121
    tmp413 = (tmp127 / tmp339)
    tmp414 = tmp413 + tmp341
    tmp415 = libdevice.rsqrt(tmp414)
    tmp416 = tmp412 * tmp415
    tmp417 = tmp416 * tmp345
    tmp418 = tmp417 + tmp347
    tmp419 = tmp92 - tmp100
    tmp420 = (tmp106 / tmp339)
    tmp421 = tmp420 + tmp341
    tmp422 = libdevice.rsqrt(tmp421)
    tmp423 = tmp419 * tmp422
    tmp424 = tmp423 * tmp345
    tmp425 = tmp424 + tmp347
    tmp426 = tmp71 - tmp79
    tmp427 = (tmp85 / tmp339)
    tmp428 = tmp427 + tmp341
    tmp429 = libdevice.rsqrt(tmp428)
    tmp430 = tmp426 * tmp429
    tmp431 = tmp430 * tmp345
    tmp432 = tmp431 + tmp347
    tmp433 = tmp50 - tmp58
    tmp434 = (tmp64 / tmp339)
    tmp435 = tmp434 + tmp341
    tmp436 = libdevice.rsqrt(tmp435)
    tmp437 = tmp433 * tmp436
    tmp438 = tmp437 * tmp345
    tmp439 = tmp438 + tmp347
    tmp440 = tmp29 - tmp37
    tmp441 = (tmp43 / tmp339)
    tmp442 = tmp441 + tmp341
    tmp443 = libdevice.rsqrt(tmp442)
    tmp444 = tmp440 * tmp443
    tmp445 = tmp444 * tmp345
    tmp446 = tmp445 + tmp347
    tmp447 = tmp6 - tmp16
    tmp448 = (tmp22 / tmp339)
    tmp449 = tmp448 + tmp341
    tmp450 = libdevice.rsqrt(tmp449)
    tmp451 = tmp447 * tmp450
    tmp452 = tmp451 * tmp345
    tmp453 = tmp452 + tmp347
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp348, r0_mask & xmask)
    tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp355, r0_mask & xmask)
    tl.store(in_out_ptr2 + (r0_1 + 768*x0), tmp362, r0_mask & xmask)
    tl.store(in_out_ptr3 + (r0_1 + 768*x0), tmp369, r0_mask & xmask)
    tl.store(in_out_ptr4 + (r0_1 + 768*x0), tmp376, r0_mask & xmask)
    tl.store(in_out_ptr5 + (r0_1 + 768*x0), tmp383, r0_mask & xmask)
    tl.store(in_out_ptr6 + (r0_1 + 768*x0), tmp390, r0_mask & xmask)
    tl.store(in_out_ptr7 + (r0_1 + 768*x0), tmp397, r0_mask & xmask)
    tl.store(in_out_ptr8 + (r0_1 + 768*x0), tmp404, r0_mask & xmask)
    tl.store(in_out_ptr9 + (r0_1 + 768*x0), tmp411, r0_mask & xmask)
    tl.store(in_out_ptr10 + (r0_1 + 768*x0), tmp418, r0_mask & xmask)
    tl.store(in_out_ptr11 + (r0_1 + 768*x0), tmp425, r0_mask & xmask)
    tl.store(in_out_ptr12 + (r0_1 + 768*x0), tmp432, r0_mask & xmask)
    tl.store(in_out_ptr13 + (r0_1 + 768*x0), tmp439, r0_mask & xmask)
    tl.store(in_out_ptr14 + (r0_1 + 768*x0), tmp446, r0_mask & xmask)
    tl.store(in_out_ptr15 + (r0_1 + 768*x0), tmp453, r0_mask & xmask)
