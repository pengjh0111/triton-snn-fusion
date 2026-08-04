
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 104448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    x2 = xindex
    tmp26 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 3, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tmp8 & tmp3
    tmp10 = tl.full([1], 0.0, tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp4 >= tmp7
    tmp14 = tl.full([1], 4, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = tmp13 & tmp3
    tmp17 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp3, tmp18, tmp19)
    tmp21 = tmp0 >= tmp2
    tmp22 = tl.full([1], 4, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp21 & xmask, other=0.0)
    tmp25 = tl.where(tmp3, tmp20, tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1], 1, tl.int64)
    tmp29 = tmp28 >= tmp0
    tmp30 = tmp28 < tmp2
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 3, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp30
    tmp37 = tl.full([1], 0.0, tl.float32)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp31 >= tmp34
    tmp41 = tl.full([1], 4, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp30
    tmp44 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp43 & xmask, other=0.0)
    tmp45 = tl.where(tmp35, tmp39, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp30, tmp45, tmp46)
    tmp48 = tmp28 >= tmp2
    tmp49 = tmp28 < tmp22
    tmp50 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp48 & xmask, other=0.0)
    tmp51 = tl.where(tmp30, tmp47, tmp50)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp27 + tmp53
    tmp55 = tl.full([1], 2, tl.int64)
    tmp56 = tmp55 >= tmp0
    tmp57 = tmp55 < tmp2
    tmp58 = tl.full([1], 3, tl.int64)
    tmp59 = tl.full([1], 0, tl.int64)
    tmp60 = tmp58 >= tmp59
    tmp61 = tmp58 < tmp58
    tmp62 = tmp61 & tmp57
    tmp63 = tl.full([1], 0.0, tl.float32)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp62, tmp63, tmp64)
    tmp66 = tmp58 >= tmp58
    tmp67 = tl.full([1], 4, tl.int64)
    tmp68 = tmp58 < tmp67
    tmp69 = tmp66 & tmp57
    tmp70 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp69 & xmask, other=0.0)
    tmp71 = tl.where(tmp61, tmp65, tmp70)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp57, tmp71, tmp72)
    tmp74 = tmp55 >= tmp2
    tmp75 = tmp55 < tmp22
    tmp76 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp74 & xmask, other=0.0)
    tmp77 = tl.where(tmp57, tmp73, tmp76)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp54 + tmp79
    tmp81 = tmp2 >= tmp0
    tmp82 = tmp2 < tmp2
    tmp83 = tl.full([1], 4, tl.int64)
    tmp84 = tl.full([1], 0, tl.int64)
    tmp85 = tmp83 >= tmp84
    tmp86 = tl.full([1], 3, tl.int64)
    tmp87 = tmp83 < tmp86
    tmp88 = tmp87 & tmp82
    tmp89 = tl.full([1], 0.0, tl.float32)
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tmp83 >= tmp86
    tmp93 = tmp83 < tmp83
    tmp94 = tmp92 & tmp82
    tmp95 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp94 & xmask, other=0.0)
    tmp96 = tl.where(tmp87, tmp91, tmp95)
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp82, tmp96, tmp97)
    tmp99 = tmp2 >= tmp2
    tmp100 = tmp2 < tmp22
    tmp101 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp99 & xmask, other=0.0)
    tmp102 = tl.where(tmp82, tmp98, tmp101)
    tmp104 = tmp102 * tmp103
    tmp105 = tmp80 + tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = -tmp107
    tmp109 = libdevice.exp(tmp108)
    tmp110 = tl.full([1], 1.0, tl.float32)
    tmp111 = tmp109 + tmp110
    tmp112 = (tmp107 / tmp111)
    tl.store(in_out_ptr0 + (x2), tmp112, xmask)
