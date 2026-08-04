
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 22, 'num_store': 8, 'num_reduction': 16, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 448512}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr12, out_ptr13, out_ptr14, out_ptr15, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_ptr4 + (46080 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr0 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr1 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (43008 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr0 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr2 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr4 + (39936 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr0 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr2 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr3 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp34 = tl.load(in_ptr4 + (36864 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp101 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp103 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp38 = tl.where(r0_mask & xmask, tmp36, 0)
    tmp39 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask & xmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp43 = tl.full([1, 1], 768, tl.int32)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = (tmp42 / tmp44)
    tmp46 = tmp36 - tmp45
    tmp47 = tmp46 * tmp46
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask & xmask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp52 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp54 = tl.where(r0_mask & xmask, tmp52, 0)
    tmp55 = tl.broadcast_to(tmp52, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask & xmask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tmp59 = (tmp58 / tmp44)
    tmp60 = tmp52 - tmp59
    tmp61 = tmp60 * tmp60
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])
    tmp64 = tl.where(r0_mask & xmask, tmp62, 0)
    tmp65 = tl.sum(tmp64, 1)[:, None].to(tl.float32)
    tmp66 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(r0_mask & xmask, tmp66, 0)
    tmp69 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    tmp71 = tl.where(r0_mask & xmask, tmp69, 0)
    tmp72 = tl.sum(tmp71, 1)[:, None].to(tl.float32)
    tmp73 = (tmp72 / tmp44)
    tmp74 = tmp66 - tmp73
    tmp75 = tmp74 * tmp74
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK, R0_BLOCK])
    tmp78 = tl.where(r0_mask & xmask, tmp76, 0)
    tmp79 = tl.sum(tmp78, 1)[:, None].to(tl.float32)
    tmp80 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
    tmp82 = tl.where(r0_mask & xmask, tmp80, 0)
    tmp83 = tl.broadcast_to(tmp80, [XBLOCK, R0_BLOCK])
    tmp85 = tl.where(r0_mask & xmask, tmp83, 0)
    tmp86 = tl.sum(tmp85, 1)[:, None].to(tl.float32)
    tmp87 = (tmp86 / tmp44)
    tmp88 = tmp80 - tmp87
    tmp89 = tmp88 * tmp88
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
    tmp92 = tl.where(r0_mask & xmask, tmp90, 0)
    tmp93 = tl.sum(tmp92, 1)[:, None].to(tl.float32)
    tmp94 = tmp35 - tmp87
    tmp95 = tl.full([1, 1], 768.0, tl.float32)
    tmp96 = (tmp93 / tmp95)
    tmp97 = tl.full([1, 1], 1e-05, tl.float32)
    tmp98 = tmp96 + tmp97
    tmp99 = libdevice.rsqrt(tmp98)
    tmp100 = tmp94 * tmp99
    tmp102 = tmp100 * tmp101
    tmp104 = tmp102 + tmp103
    tmp105 = tmp26 - tmp73
    tmp106 = (tmp79 / tmp95)
    tmp107 = tmp106 + tmp97
    tmp108 = libdevice.rsqrt(tmp107)
    tmp109 = tmp105 * tmp108
    tmp110 = tmp109 * tmp101
    tmp111 = tmp110 + tmp103
    tmp112 = tmp17 - tmp59
    tmp113 = (tmp65 / tmp95)
    tmp114 = tmp113 + tmp97
    tmp115 = libdevice.rsqrt(tmp114)
    tmp116 = tmp112 * tmp115
    tmp117 = tmp116 * tmp101
    tmp118 = tmp117 + tmp103
    tmp119 = tmp8 - tmp45
    tmp120 = (tmp51 / tmp95)
    tmp121 = tmp120 + tmp97
    tmp122 = libdevice.rsqrt(tmp121)
    tmp123 = tmp119 * tmp122
    tmp124 = tmp123 * tmp101
    tmp125 = tmp124 + tmp103
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp17, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp26, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
    tl.store(out_ptr12 + (r0_1 + 768*x0), tmp104, r0_mask & xmask)
    tl.store(out_ptr13 + (r0_1 + 768*x0), tmp111, r0_mask & xmask)
    tl.store(out_ptr14 + (r0_1 + 768*x0), tmp118, r0_mask & xmask)
    tl.store(out_ptr15 + (r0_1 + 768*x0), tmp125, r0_mask & xmask)
