
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 153600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_25(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    x2 = xindex
    tmp12 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 3, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1], 4, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp11 = tl.where(tmp3, tmp6, tmp10)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp14 >= tmp0
    tmp16 = tmp14 < tmp2
    tmp17 = tl.full([1], 0.0, tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tmp14 >= tmp2
    tmp21 = tmp14 < tmp8
    tmp22 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp23 = tl.where(tmp16, tmp19, tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = tl.full([1], 2, tl.int64)
    tmp28 = tmp27 >= tmp0
    tmp29 = tmp27 < tmp2
    tmp30 = tl.full([1], 0.0, tl.float32)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp27 >= tmp2
    tmp34 = tmp27 < tmp8
    tmp35 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp36 = tl.where(tmp29, tmp32, tmp35)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp26 + tmp38
    tmp40 = tmp2 >= tmp0
    tmp41 = tmp2 < tmp2
    tmp42 = tl.full([1], 0.0, tl.float32)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp2 >= tmp2
    tmp46 = tmp2 < tmp8
    tmp47 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp48 = tl.where(tmp41, tmp44, tmp47)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp39 + tmp50
    tmp52 = tl.full([1], 1, tl.int64)
    tmp53 = tl.full([1], 0, tl.int64)
    tmp54 = tmp52 >= tmp53
    tmp55 = tl.full([1], 3, tl.int64)
    tmp56 = tmp52 < tmp55
    tmp57 = tmp56 & tmp3
    tmp58 = tl.full([1], 0.0, tl.float32)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp57, tmp58, tmp59)
    tmp61 = tmp52 >= tmp55
    tmp62 = tl.full([1], 4, tl.int64)
    tmp63 = tmp52 < tmp62
    tmp64 = tmp61 & tmp3
    tmp65 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp66 = tl.where(tmp56, tmp60, tmp65)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp3, tmp66, tmp67)
    tmp69 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp70 = tl.where(tmp3, tmp68, tmp69)
    tmp71 = tmp70 * tmp12
    tmp72 = tl.full([1], 2, tl.int64)
    tmp73 = tl.full([1], 0, tl.int64)
    tmp74 = tmp72 >= tmp73
    tmp75 = tl.full([1], 3, tl.int64)
    tmp76 = tmp72 < tmp75
    tmp77 = tmp76 & tmp16
    tmp78 = tl.full([1], 0.0, tl.float32)
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp77, tmp78, tmp79)
    tmp81 = tmp72 >= tmp75
    tmp82 = tl.full([1], 4, tl.int64)
    tmp83 = tmp72 < tmp82
    tmp84 = tmp81 & tmp16
    tmp85 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp86 = tl.where(tmp76, tmp80, tmp85)
    tmp87 = tl.full(tmp86.shape, 0.0, tmp86.dtype)
    tmp88 = tl.where(tmp16, tmp86, tmp87)
    tmp89 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp90 = tl.where(tmp16, tmp88, tmp89)
    tmp91 = tmp90 * tmp24
    tmp92 = tmp71 + tmp91
    tmp93 = tl.full([1], 3, tl.int64)
    tmp94 = tl.full([1], 0, tl.int64)
    tmp95 = tmp93 >= tmp94
    tmp96 = tmp93 < tmp93
    tmp97 = tmp96 & tmp29
    tmp98 = tl.full([1], 0.0, tl.float32)
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp97, tmp98, tmp99)
    tmp101 = tmp93 >= tmp93
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp93 < tmp102
    tmp104 = tmp101 & tmp29
    tmp105 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp106 = tl.where(tmp96, tmp100, tmp105)
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp29, tmp106, tmp107)
    tmp109 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp110 = tl.where(tmp29, tmp108, tmp109)
    tmp111 = tmp110 * tmp37
    tmp112 = tmp92 + tmp111
    tmp113 = tl.full([1], 4, tl.int64)
    tmp114 = tl.full([1], 0, tl.int64)
    tmp115 = tmp113 >= tmp114
    tmp116 = tl.full([1], 3, tl.int64)
    tmp117 = tmp113 < tmp116
    tmp118 = tmp117 & tmp41
    tmp119 = tl.full([1], 0.0, tl.float32)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp118, tmp119, tmp120)
    tmp122 = tmp113 >= tmp116
    tmp123 = tmp113 < tmp113
    tmp124 = tmp122 & tmp41
    tmp125 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp126 = tl.where(tmp117, tmp121, tmp125)
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp41, tmp126, tmp127)
    tmp129 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp130 = tl.where(tmp41, tmp128, tmp129)
    tmp131 = tmp130 * tmp49
    tmp132 = tmp112 + tmp131
    tmp134 = tmp51 + tmp133
    tmp135 = -tmp134
    tmp136 = libdevice.exp(tmp135)
    tmp137 = tl.full([1], 1.0, tl.float32)
    tmp138 = tmp136 + tmp137
    tmp139 = (tmp134 / tmp138)
    tmp140 = tmp132 + tmp133
    tmp141 = -tmp140
    tmp142 = libdevice.exp(tmp141)
    tmp143 = tmp142 + tmp137
    tmp144 = (tmp140 / tmp143)
    tl.store(in_out_ptr0 + (x2), tmp139, xmask)
    tl.store(in_out_ptr1 + (x2), tmp144, xmask)
