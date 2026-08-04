
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'in_ptr42': '*fp32', 'in_ptr43': '*fp32', 'in_ptr44': '*fp32', 'in_ptr45': '*fp32', 'in_ptr46': '*fp32', 'in_ptr47': '*fp32', 'in_ptr48': '*fp32', 'in_ptr49': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'out_ptr41': '*fp32', 'out_ptr42': '*fp32', 'out_ptr43': '*fp32', 'out_ptr44': '*fp32', 'out_ptr45': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]], (31,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (33,): [['tt.divisibility', 16]], (34,): [['tt.divisibility', 16]], (35,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (37,): [['tt.divisibility', 16]], (38,): [['tt.divisibility', 16]], (39,): [['tt.divisibility', 16]], (40,): [['tt.divisibility', 16]], (41,): [['tt.divisibility', 16]], (42,): [['tt.divisibility', 16]], (43,): [['tt.divisibility', 16]], (44,): [['tt.divisibility', 16]], (45,): [['tt.divisibility', 16]], (46,): [['tt.divisibility', 16]], (47,): [['tt.divisibility', 16]], (48,): [['tt.divisibility', 16]], (49,): [['tt.divisibility', 16]], (50,): [['tt.divisibility', 16]], (51,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (53,): [['tt.divisibility', 16]], (54,): [['tt.divisibility', 16]], (55,): [['tt.divisibility', 16]], (56,): [['tt.divisibility', 16]], (57,): [['tt.divisibility', 16]], (58,): [['tt.divisibility', 16]], (59,): [['tt.divisibility', 16]], (60,): [['tt.divisibility', 16]], (61,): [['tt.divisibility', 16]], (62,): [['tt.divisibility', 16]], (63,): [['tt.divisibility', 16]], (64,): [['tt.divisibility', 16]], (65,): [['tt.divisibility', 16]], (66,): [['tt.divisibility', 16]], (67,): [['tt.divisibility', 16]], (68,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_exp_mul_softplus_split_with_sizes_sum_unsqueeze_zeros_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 67, 'num_store': 16, 'num_reduction': 16, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1585152, 'r0_': 106496}}
)
@triton.jit
def triton_per_fused_add_addmm_exp_mul_softplus_split_with_sizes_sum_unsqueeze_zeros_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    x0 = (xindex % 1536)
    x3 = xindex
    r0_2 = r0_index
    x1 = xindex // 1536
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (r0_2 + 16*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr10 + (x3), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr12 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr13 + (x3), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr14 + (x3), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr15 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr16 + (x3), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr17 + (x3), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr18 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tl.load(in_ptr19 + (x3), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr20 + (x3), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr21 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp99 = tl.load(in_ptr22 + (x3), xmask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr23 + (x3), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr24 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tl.load(in_ptr25 + (x3), xmask, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr26 + (x3), xmask, eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr27 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp127 = tl.load(in_ptr28 + (x3), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr29 + (x3), xmask, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr30 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp141 = tl.load(in_ptr31 + (x3), xmask, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr32 + (x3), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr33 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp155 = tl.load(in_ptr34 + (x3), xmask, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr35 + (x3), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr36 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp169 = tl.load(in_ptr37 + (x3), xmask, eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr38 + (x3), xmask, eviction_policy='evict_last')
    tmp181 = tl.load(in_ptr39 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp183 = tl.load(in_ptr40 + (x3), xmask, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr41 + (x3), xmask, eviction_policy='evict_last')
    tmp195 = tl.load(in_ptr42 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp197 = tl.load(in_ptr43 + (x3), xmask, eviction_policy='evict_last')
    tmp200 = tl.load(in_ptr44 + (x3), xmask, eviction_policy='evict_last')
    tmp209 = tl.load(in_ptr45 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp211 = tl.load(in_ptr46 + (x3), xmask, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr45 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp220 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr47 + (48 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp231 = tl.load(in_ptr48 + (x3), xmask, eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr47 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp240 = tl.load(in_ptr3 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp246 = tl.load(in_ptr6 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp252 = tl.load(in_ptr9 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp258 = tl.load(in_ptr12 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp264 = tl.load(in_ptr15 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp270 = tl.load(in_ptr18 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp276 = tl.load(in_ptr21 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp282 = tl.load(in_ptr24 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp288 = tl.load(in_ptr27 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp294 = tl.load(in_ptr30 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp300 = tl.load(in_ptr33 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp306 = tl.load(in_ptr36 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp312 = tl.load(in_ptr39 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp318 = tl.load(in_ptr42 + (64 + r0_2 + 80*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp324 = tl.load(in_ptr49 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = libdevice.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tl.full([1, 1], 0.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp14 = tmp7 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 + tmp16
    tmp19 = tmp0 + tmp18
    tmp20 = tmp19 > tmp3
    tmp21 = libdevice.exp(tmp19)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp19, tmp22)
    tmp24 = tmp23 * tmp8
    tmp25 = libdevice.exp(tmp24)
    tmp26 = tmp25 * tmp17
    tmp28 = tmp23 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp26 + tmp30
    tmp33 = tmp0 + tmp32
    tmp34 = tmp33 > tmp3
    tmp35 = libdevice.exp(tmp33)
    tmp36 = libdevice.log1p(tmp35)
    tmp37 = tl.where(tmp34, tmp33, tmp36)
    tmp38 = tmp37 * tmp8
    tmp39 = libdevice.exp(tmp38)
    tmp40 = tmp39 * tmp31
    tmp42 = tmp37 * tmp41
    tmp44 = tmp42 * tmp43
    tmp45 = tmp40 + tmp44
    tmp47 = tmp0 + tmp46
    tmp48 = tmp47 > tmp3
    tmp49 = libdevice.exp(tmp47)
    tmp50 = libdevice.log1p(tmp49)
    tmp51 = tl.where(tmp48, tmp47, tmp50)
    tmp52 = tmp51 * tmp8
    tmp53 = libdevice.exp(tmp52)
    tmp54 = tmp53 * tmp45
    tmp56 = tmp51 * tmp55
    tmp58 = tmp56 * tmp57
    tmp59 = tmp54 + tmp58
    tmp61 = tmp0 + tmp60
    tmp62 = tmp61 > tmp3
    tmp63 = libdevice.exp(tmp61)
    tmp64 = libdevice.log1p(tmp63)
    tmp65 = tl.where(tmp62, tmp61, tmp64)
    tmp66 = tmp65 * tmp8
    tmp67 = libdevice.exp(tmp66)
    tmp68 = tmp67 * tmp59
    tmp70 = tmp65 * tmp69
    tmp72 = tmp70 * tmp71
    tmp73 = tmp68 + tmp72
    tmp75 = tmp0 + tmp74
    tmp76 = tmp75 > tmp3
    tmp77 = libdevice.exp(tmp75)
    tmp78 = libdevice.log1p(tmp77)
    tmp79 = tl.where(tmp76, tmp75, tmp78)
    tmp80 = tmp79 * tmp8
    tmp81 = libdevice.exp(tmp80)
    tmp82 = tmp81 * tmp73
    tmp84 = tmp79 * tmp83
    tmp86 = tmp84 * tmp85
    tmp87 = tmp82 + tmp86
    tmp89 = tmp0 + tmp88
    tmp90 = tmp89 > tmp3
    tmp91 = libdevice.exp(tmp89)
    tmp92 = libdevice.log1p(tmp91)
    tmp93 = tl.where(tmp90, tmp89, tmp92)
    tmp94 = tmp93 * tmp8
    tmp95 = libdevice.exp(tmp94)
    tmp96 = tmp95 * tmp87
    tmp98 = tmp93 * tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tmp103 = tmp0 + tmp102
    tmp104 = tmp103 > tmp3
    tmp105 = libdevice.exp(tmp103)
    tmp106 = libdevice.log1p(tmp105)
    tmp107 = tl.where(tmp104, tmp103, tmp106)
    tmp108 = tmp107 * tmp8
    tmp109 = libdevice.exp(tmp108)
    tmp110 = tmp109 * tmp101
    tmp112 = tmp107 * tmp111
    tmp114 = tmp112 * tmp113
    tmp115 = tmp110 + tmp114
    tmp117 = tmp0 + tmp116
    tmp118 = tmp117 > tmp3
    tmp119 = libdevice.exp(tmp117)
    tmp120 = libdevice.log1p(tmp119)
    tmp121 = tl.where(tmp118, tmp117, tmp120)
    tmp122 = tmp121 * tmp8
    tmp123 = libdevice.exp(tmp122)
    tmp124 = tmp123 * tmp115
    tmp126 = tmp121 * tmp125
    tmp128 = tmp126 * tmp127
    tmp129 = tmp124 + tmp128
    tmp131 = tmp0 + tmp130
    tmp132 = tmp131 > tmp3
    tmp133 = libdevice.exp(tmp131)
    tmp134 = libdevice.log1p(tmp133)
    tmp135 = tl.where(tmp132, tmp131, tmp134)
    tmp136 = tmp135 * tmp8
    tmp137 = libdevice.exp(tmp136)
    tmp138 = tmp137 * tmp129
    tmp140 = tmp135 * tmp139
    tmp142 = tmp140 * tmp141
    tmp143 = tmp138 + tmp142
    tmp145 = tmp0 + tmp144
    tmp146 = tmp145 > tmp3
    tmp147 = libdevice.exp(tmp145)
    tmp148 = libdevice.log1p(tmp147)
    tmp149 = tl.where(tmp146, tmp145, tmp148)
    tmp150 = tmp149 * tmp8
    tmp151 = libdevice.exp(tmp150)
    tmp152 = tmp151 * tmp143
    tmp154 = tmp149 * tmp153
    tmp156 = tmp154 * tmp155
    tmp157 = tmp152 + tmp156
    tmp159 = tmp0 + tmp158
    tmp160 = tmp159 > tmp3
    tmp161 = libdevice.exp(tmp159)
    tmp162 = libdevice.log1p(tmp161)
    tmp163 = tl.where(tmp160, tmp159, tmp162)
    tmp164 = tmp163 * tmp8
    tmp165 = libdevice.exp(tmp164)
    tmp166 = tmp165 * tmp157
    tmp168 = tmp163 * tmp167
    tmp170 = tmp168 * tmp169
    tmp171 = tmp166 + tmp170
    tmp173 = tmp0 + tmp172
    tmp174 = tmp173 > tmp3
    tmp175 = libdevice.exp(tmp173)
    tmp176 = libdevice.log1p(tmp175)
    tmp177 = tl.where(tmp174, tmp173, tmp176)
    tmp178 = tmp177 * tmp8
    tmp179 = libdevice.exp(tmp178)
    tmp180 = tmp179 * tmp171
    tmp182 = tmp177 * tmp181
    tmp184 = tmp182 * tmp183
    tmp185 = tmp180 + tmp184
    tmp187 = tmp0 + tmp186
    tmp188 = tmp187 > tmp3
    tmp189 = libdevice.exp(tmp187)
    tmp190 = libdevice.log1p(tmp189)
    tmp191 = tl.where(tmp188, tmp187, tmp190)
    tmp192 = tmp191 * tmp8
    tmp193 = libdevice.exp(tmp192)
    tmp194 = tmp193 * tmp185
    tmp196 = tmp191 * tmp195
    tmp198 = tmp196 * tmp197
    tmp199 = tmp194 + tmp198
    tmp201 = tmp0 + tmp200
    tmp202 = tmp201 > tmp3
    tmp203 = libdevice.exp(tmp201)
    tmp204 = libdevice.log1p(tmp203)
    tmp205 = tl.where(tmp202, tmp201, tmp204)
    tmp206 = tmp205 * tmp8
    tmp207 = libdevice.exp(tmp206)
    tmp208 = tmp207 * tmp199
    tmp210 = tmp205 * tmp209
    tmp212 = tmp210 * tmp211
    tmp213 = tmp208 + tmp212
    tmp215 = tmp213 * tmp214
    tmp216 = tl.broadcast_to(tmp215, [XBLOCK, R0_BLOCK])
    tmp218 = tl.where(r0_mask & xmask, tmp216, 0)
    tmp219 = tl.sum(tmp218, 1)[:, None].to(tl.float32)
    tmp221 = tmp0 + tmp220
    tmp222 = tmp221 > tmp3
    tmp223 = libdevice.exp(tmp221)
    tmp224 = libdevice.log1p(tmp223)
    tmp225 = tl.where(tmp222, tmp221, tmp224)
    tmp226 = tmp225 * tmp8
    tmp227 = libdevice.exp(tmp226)
    tmp228 = tmp227 * tmp213
    tmp230 = tmp225 * tmp229
    tmp232 = tmp230 * tmp231
    tmp233 = tmp228 + tmp232
    tmp235 = tmp233 * tmp234
    tmp236 = tl.broadcast_to(tmp235, [XBLOCK, R0_BLOCK])
    tmp238 = tl.where(r0_mask & xmask, tmp236, 0)
    tmp239 = tl.sum(tmp238, 1)[:, None].to(tl.float32)
    tmp241 = tmp17 * tmp240
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK, R0_BLOCK])
    tmp244 = tl.where(r0_mask & xmask, tmp242, 0)
    tmp245 = tl.sum(tmp244, 1)[:, None].to(tl.float32)
    tmp247 = tmp31 * tmp246
    tmp248 = tl.broadcast_to(tmp247, [XBLOCK, R0_BLOCK])
    tmp250 = tl.where(r0_mask & xmask, tmp248, 0)
    tmp251 = tl.sum(tmp250, 1)[:, None].to(tl.float32)
    tmp253 = tmp45 * tmp252
    tmp254 = tl.broadcast_to(tmp253, [XBLOCK, R0_BLOCK])
    tmp256 = tl.where(r0_mask & xmask, tmp254, 0)
    tmp257 = tl.sum(tmp256, 1)[:, None].to(tl.float32)
    tmp259 = tmp59 * tmp258
    tmp260 = tl.broadcast_to(tmp259, [XBLOCK, R0_BLOCK])
    tmp262 = tl.where(r0_mask & xmask, tmp260, 0)
    tmp263 = tl.sum(tmp262, 1)[:, None].to(tl.float32)
    tmp265 = tmp73 * tmp264
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, R0_BLOCK])
    tmp268 = tl.where(r0_mask & xmask, tmp266, 0)
    tmp269 = tl.sum(tmp268, 1)[:, None].to(tl.float32)
    tmp271 = tmp87 * tmp270
    tmp272 = tl.broadcast_to(tmp271, [XBLOCK, R0_BLOCK])
    tmp274 = tl.where(r0_mask & xmask, tmp272, 0)
    tmp275 = tl.sum(tmp274, 1)[:, None].to(tl.float32)
    tmp277 = tmp101 * tmp276
    tmp278 = tl.broadcast_to(tmp277, [XBLOCK, R0_BLOCK])
    tmp280 = tl.where(r0_mask & xmask, tmp278, 0)
    tmp281 = tl.sum(tmp280, 1)[:, None].to(tl.float32)
    tmp283 = tmp115 * tmp282
    tmp284 = tl.broadcast_to(tmp283, [XBLOCK, R0_BLOCK])
    tmp286 = tl.where(r0_mask & xmask, tmp284, 0)
    tmp287 = tl.sum(tmp286, 1)[:, None].to(tl.float32)
    tmp289 = tmp129 * tmp288
    tmp290 = tl.broadcast_to(tmp289, [XBLOCK, R0_BLOCK])
    tmp292 = tl.where(r0_mask & xmask, tmp290, 0)
    tmp293 = tl.sum(tmp292, 1)[:, None].to(tl.float32)
    tmp295 = tmp143 * tmp294
    tmp296 = tl.broadcast_to(tmp295, [XBLOCK, R0_BLOCK])
    tmp298 = tl.where(r0_mask & xmask, tmp296, 0)
    tmp299 = tl.sum(tmp298, 1)[:, None].to(tl.float32)
    tmp301 = tmp157 * tmp300
    tmp302 = tl.broadcast_to(tmp301, [XBLOCK, R0_BLOCK])
    tmp304 = tl.where(r0_mask & xmask, tmp302, 0)
    tmp305 = tl.sum(tmp304, 1)[:, None].to(tl.float32)
    tmp307 = tmp171 * tmp306
    tmp308 = tl.broadcast_to(tmp307, [XBLOCK, R0_BLOCK])
    tmp310 = tl.where(r0_mask & xmask, tmp308, 0)
    tmp311 = tl.sum(tmp310, 1)[:, None].to(tl.float32)
    tmp313 = tmp185 * tmp312
    tmp314 = tl.broadcast_to(tmp313, [XBLOCK, R0_BLOCK])
    tmp316 = tl.where(r0_mask & xmask, tmp314, 0)
    tmp317 = tl.sum(tmp316, 1)[:, None].to(tl.float32)
    tmp319 = tmp199 * tmp318
    tmp320 = tl.broadcast_to(tmp319, [XBLOCK, R0_BLOCK])
    tmp322 = tl.where(r0_mask & xmask, tmp320, 0)
    tmp323 = tl.sum(tmp322, 1)[:, None].to(tl.float32)
    tmp325 = tmp324 * tmp15
    tmp326 = tmp245 + tmp325
    tmp327 = tmp324 * tmp29
    tmp328 = tmp251 + tmp327
    tmp329 = tmp324 * tmp43
    tmp330 = tmp257 + tmp329
    tmp331 = tmp324 * tmp57
    tmp332 = tmp263 + tmp331
    tmp333 = tmp324 * tmp71
    tmp334 = tmp269 + tmp333
    tmp335 = tmp324 * tmp85
    tmp336 = tmp275 + tmp335
    tmp337 = tmp324 * tmp99
    tmp338 = tmp281 + tmp337
    tmp339 = tmp324 * tmp113
    tmp340 = tmp287 + tmp339
    tmp341 = tmp324 * tmp127
    tmp342 = tmp293 + tmp341
    tmp343 = tmp324 * tmp141
    tmp344 = tmp299 + tmp343
    tmp345 = tmp324 * tmp155
    tmp346 = tmp305 + tmp345
    tmp347 = tmp324 * tmp169
    tmp348 = tmp311 + tmp347
    tmp349 = tmp324 * tmp183
    tmp350 = tmp317 + tmp349
    tmp351 = tmp324 * tmp197
    tmp352 = tmp323 + tmp351
    tmp353 = tmp324 * tmp211
    tmp354 = tmp219 + tmp353
    tmp355 = tmp324 * tmp231
    tmp356 = tmp239 + tmp355
    tl.store(out_ptr30 + (x3), tmp326, xmask)
    tl.store(out_ptr31 + (x3), tmp328, xmask)
    tl.store(out_ptr32 + (x3), tmp330, xmask)
    tl.store(out_ptr33 + (x3), tmp332, xmask)
    tl.store(out_ptr34 + (x3), tmp334, xmask)
    tl.store(out_ptr35 + (x3), tmp336, xmask)
    tl.store(out_ptr36 + (x3), tmp338, xmask)
    tl.store(out_ptr37 + (x3), tmp340, xmask)
    tl.store(out_ptr38 + (x3), tmp342, xmask)
    tl.store(out_ptr39 + (x3), tmp344, xmask)
    tl.store(out_ptr40 + (x3), tmp346, xmask)
    tl.store(out_ptr41 + (x3), tmp348, xmask)
    tl.store(out_ptr42 + (x3), tmp350, xmask)
    tl.store(out_ptr43 + (x3), tmp352, xmask)
    tl.store(out_ptr44 + (x3), tmp354, xmask)
    tl.store(out_ptr45 + (x3), tmp356, xmask)
