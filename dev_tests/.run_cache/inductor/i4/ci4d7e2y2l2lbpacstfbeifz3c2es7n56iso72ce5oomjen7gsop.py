
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'in_ptr42': '*fp32', 'in_ptr43': '*fp32', 'in_ptr44': '*fp32', 'in_ptr45': '*fp32', 'in_ptr46': '*fp32', 'in_ptr47': '*fp32', 'in_ptr48': '*fp32', 'in_ptr49': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]], (31,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (33,): [['tt.divisibility', 16]], (34,): [['tt.divisibility', 16]], (35,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (37,): [['tt.divisibility', 16]], (38,): [['tt.divisibility', 16]], (39,): [['tt.divisibility', 16]], (40,): [['tt.divisibility', 16]], (41,): [['tt.divisibility', 16]], (42,): [['tt.divisibility', 16]], (43,): [['tt.divisibility', 16]], (44,): [['tt.divisibility', 16]], (45,): [['tt.divisibility', 16]], (46,): [['tt.divisibility', 16]], (47,): [['tt.divisibility', 16]], (48,): [['tt.divisibility', 16]], (49,): [['tt.divisibility', 16]], (50,): [['tt.divisibility', 16]], (51,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (53,): [['tt.divisibility', 16]], (54,): [['tt.divisibility', 16]], (55,): [['tt.divisibility', 16]], (56,): [['tt.divisibility', 16]], (57,): [['tt.divisibility', 16]], (58,): [['tt.divisibility', 16]], (59,): [['tt.divisibility', 16]], (60,): [['tt.divisibility', 16]], (61,): [['tt.divisibility', 16]], (62,): [['tt.divisibility', 16]], (63,): [['tt.divisibility', 16]], (64,): [['tt.divisibility', 16]], (65,): [['tt.divisibility', 16]], (66,): [['tt.divisibility', 16]], (67,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 50, 'num_store': 16, 'num_reduction': 16, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 17041408, 'r0_': 131072}}
)
@triton.jit
def triton_red_fused_mean_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp54 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp67 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp80 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp93 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp106 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp119 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp132 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp145 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp158 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp171 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp184 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp197 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp210 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr5 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr6 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr7 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr8 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr9 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr10 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.load(in_ptr11 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tl.load(in_ptr12 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr13 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr14 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp57 = tl.load(in_ptr15 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tl.load(in_ptr16 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr17 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp70 = tl.load(in_ptr18 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp72 = tl.load(in_ptr19 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp82 = tl.load(in_ptr20 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp83 = tl.load(in_ptr21 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp85 = tl.load(in_ptr22 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp95 = tl.load(in_ptr23 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp96 = tl.load(in_ptr24 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp98 = tl.load(in_ptr25 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp108 = tl.load(in_ptr26 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp109 = tl.load(in_ptr27 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp111 = tl.load(in_ptr28 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp121 = tl.load(in_ptr29 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp122 = tl.load(in_ptr30 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp124 = tl.load(in_ptr31 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp134 = tl.load(in_ptr32 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp135 = tl.load(in_ptr33 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp137 = tl.load(in_ptr34 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp147 = tl.load(in_ptr35 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp148 = tl.load(in_ptr36 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.load(in_ptr37 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp160 = tl.load(in_ptr38 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp161 = tl.load(in_ptr39 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp163 = tl.load(in_ptr40 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp173 = tl.load(in_ptr41 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp174 = tl.load(in_ptr42 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp176 = tl.load(in_ptr43 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp186 = tl.load(in_ptr44 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp187 = tl.load(in_ptr45 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp189 = tl.load(in_ptr46 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp199 = tl.load(in_ptr47 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp200 = tl.load(in_ptr48 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp202 = tl.load(in_ptr49 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 256.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp21 = (tmp20 / tmp4)
        tmp22 = tmp21 + tmp6
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24 * tmp10
        tmp26 = tmp25 + tmp12
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
        tmp32 = tmp30 - tmp31
        tmp34 = (tmp33 / tmp4)
        tmp35 = tmp34 + tmp6
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp32 * tmp36
        tmp38 = tmp37 * tmp10
        tmp39 = tmp38 + tmp12
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
        tmp45 = tmp43 - tmp44
        tmp47 = (tmp46 / tmp4)
        tmp48 = tmp47 + tmp6
        tmp49 = libdevice.rsqrt(tmp48)
        tmp50 = tmp45 * tmp49
        tmp51 = tmp50 * tmp10
        tmp52 = tmp51 + tmp12
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK, R0_BLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(r0_mask & xmask, tmp55, _tmp54)
        tmp58 = tmp56 - tmp57
        tmp60 = (tmp59 / tmp4)
        tmp61 = tmp60 + tmp6
        tmp62 = libdevice.rsqrt(tmp61)
        tmp63 = tmp58 * tmp62
        tmp64 = tmp63 * tmp10
        tmp65 = tmp64 + tmp12
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(r0_mask & xmask, tmp68, _tmp67)
        tmp71 = tmp69 - tmp70
        tmp73 = (tmp72 / tmp4)
        tmp74 = tmp73 + tmp6
        tmp75 = libdevice.rsqrt(tmp74)
        tmp76 = tmp71 * tmp75
        tmp77 = tmp76 * tmp10
        tmp78 = tmp77 + tmp12
        tmp79 = tl.broadcast_to(tmp78, [XBLOCK, R0_BLOCK])
        tmp81 = _tmp80 + tmp79
        _tmp80 = tl.where(r0_mask & xmask, tmp81, _tmp80)
        tmp84 = tmp82 - tmp83
        tmp86 = (tmp85 / tmp4)
        tmp87 = tmp86 + tmp6
        tmp88 = libdevice.rsqrt(tmp87)
        tmp89 = tmp84 * tmp88
        tmp90 = tmp89 * tmp10
        tmp91 = tmp90 + tmp12
        tmp92 = tl.broadcast_to(tmp91, [XBLOCK, R0_BLOCK])
        tmp94 = _tmp93 + tmp92
        _tmp93 = tl.where(r0_mask & xmask, tmp94, _tmp93)
        tmp97 = tmp95 - tmp96
        tmp99 = (tmp98 / tmp4)
        tmp100 = tmp99 + tmp6
        tmp101 = libdevice.rsqrt(tmp100)
        tmp102 = tmp97 * tmp101
        tmp103 = tmp102 * tmp10
        tmp104 = tmp103 + tmp12
        tmp105 = tl.broadcast_to(tmp104, [XBLOCK, R0_BLOCK])
        tmp107 = _tmp106 + tmp105
        _tmp106 = tl.where(r0_mask & xmask, tmp107, _tmp106)
        tmp110 = tmp108 - tmp109
        tmp112 = (tmp111 / tmp4)
        tmp113 = tmp112 + tmp6
        tmp114 = libdevice.rsqrt(tmp113)
        tmp115 = tmp110 * tmp114
        tmp116 = tmp115 * tmp10
        tmp117 = tmp116 + tmp12
        tmp118 = tl.broadcast_to(tmp117, [XBLOCK, R0_BLOCK])
        tmp120 = _tmp119 + tmp118
        _tmp119 = tl.where(r0_mask & xmask, tmp120, _tmp119)
        tmp123 = tmp121 - tmp122
        tmp125 = (tmp124 / tmp4)
        tmp126 = tmp125 + tmp6
        tmp127 = libdevice.rsqrt(tmp126)
        tmp128 = tmp123 * tmp127
        tmp129 = tmp128 * tmp10
        tmp130 = tmp129 + tmp12
        tmp131 = tl.broadcast_to(tmp130, [XBLOCK, R0_BLOCK])
        tmp133 = _tmp132 + tmp131
        _tmp132 = tl.where(r0_mask & xmask, tmp133, _tmp132)
        tmp136 = tmp134 - tmp135
        tmp138 = (tmp137 / tmp4)
        tmp139 = tmp138 + tmp6
        tmp140 = libdevice.rsqrt(tmp139)
        tmp141 = tmp136 * tmp140
        tmp142 = tmp141 * tmp10
        tmp143 = tmp142 + tmp12
        tmp144 = tl.broadcast_to(tmp143, [XBLOCK, R0_BLOCK])
        tmp146 = _tmp145 + tmp144
        _tmp145 = tl.where(r0_mask & xmask, tmp146, _tmp145)
        tmp149 = tmp147 - tmp148
        tmp151 = (tmp150 / tmp4)
        tmp152 = tmp151 + tmp6
        tmp153 = libdevice.rsqrt(tmp152)
        tmp154 = tmp149 * tmp153
        tmp155 = tmp154 * tmp10
        tmp156 = tmp155 + tmp12
        tmp157 = tl.broadcast_to(tmp156, [XBLOCK, R0_BLOCK])
        tmp159 = _tmp158 + tmp157
        _tmp158 = tl.where(r0_mask & xmask, tmp159, _tmp158)
        tmp162 = tmp160 - tmp161
        tmp164 = (tmp163 / tmp4)
        tmp165 = tmp164 + tmp6
        tmp166 = libdevice.rsqrt(tmp165)
        tmp167 = tmp162 * tmp166
        tmp168 = tmp167 * tmp10
        tmp169 = tmp168 + tmp12
        tmp170 = tl.broadcast_to(tmp169, [XBLOCK, R0_BLOCK])
        tmp172 = _tmp171 + tmp170
        _tmp171 = tl.where(r0_mask & xmask, tmp172, _tmp171)
        tmp175 = tmp173 - tmp174
        tmp177 = (tmp176 / tmp4)
        tmp178 = tmp177 + tmp6
        tmp179 = libdevice.rsqrt(tmp178)
        tmp180 = tmp175 * tmp179
        tmp181 = tmp180 * tmp10
        tmp182 = tmp181 + tmp12
        tmp183 = tl.broadcast_to(tmp182, [XBLOCK, R0_BLOCK])
        tmp185 = _tmp184 + tmp183
        _tmp184 = tl.where(r0_mask & xmask, tmp185, _tmp184)
        tmp188 = tmp186 - tmp187
        tmp190 = (tmp189 / tmp4)
        tmp191 = tmp190 + tmp6
        tmp192 = libdevice.rsqrt(tmp191)
        tmp193 = tmp188 * tmp192
        tmp194 = tmp193 * tmp10
        tmp195 = tmp194 + tmp12
        tmp196 = tl.broadcast_to(tmp195, [XBLOCK, R0_BLOCK])
        tmp198 = _tmp197 + tmp196
        _tmp197 = tl.where(r0_mask & xmask, tmp198, _tmp197)
        tmp201 = tmp199 - tmp200
        tmp203 = (tmp202 / tmp4)
        tmp204 = tmp203 + tmp6
        tmp205 = libdevice.rsqrt(tmp204)
        tmp206 = tmp201 * tmp205
        tmp207 = tmp206 * tmp10
        tmp208 = tmp207 + tmp12
        tmp209 = tl.broadcast_to(tmp208, [XBLOCK, R0_BLOCK])
        tmp211 = _tmp210 + tmp209
        _tmp210 = tl.where(r0_mask & xmask, tmp211, _tmp210)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    tmp80 = tl.sum(_tmp80, 1)[:, None]
    tmp93 = tl.sum(_tmp93, 1)[:, None]
    tmp106 = tl.sum(_tmp106, 1)[:, None]
    tmp119 = tl.sum(_tmp119, 1)[:, None]
    tmp132 = tl.sum(_tmp132, 1)[:, None]
    tmp145 = tl.sum(_tmp145, 1)[:, None]
    tmp158 = tl.sum(_tmp158, 1)[:, None]
    tmp171 = tl.sum(_tmp171, 1)[:, None]
    tmp184 = tl.sum(_tmp184, 1)[:, None]
    tmp197 = tl.sum(_tmp197, 1)[:, None]
    tmp210 = tl.sum(_tmp210, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
    tl.store(out_ptr2 + (x3), tmp41, xmask)
    tl.store(out_ptr3 + (x3), tmp54, xmask)
    tl.store(out_ptr4 + (x3), tmp67, xmask)
    tl.store(out_ptr5 + (x3), tmp80, xmask)
    tl.store(out_ptr6 + (x3), tmp93, xmask)
    tl.store(out_ptr7 + (x3), tmp106, xmask)
    tl.store(out_ptr8 + (x3), tmp119, xmask)
    tl.store(out_ptr9 + (x3), tmp132, xmask)
    tl.store(out_ptr10 + (x3), tmp145, xmask)
    tl.store(out_ptr11 + (x3), tmp158, xmask)
    tl.store(out_ptr12 + (x3), tmp171, xmask)
    tl.store(out_ptr13 + (x3), tmp184, xmask)
    tl.store(out_ptr14 + (x3), tmp197, xmask)
    tl.store(out_ptr15 + (x3), tmp210, xmask)
