
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 79872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp52 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp53 = tmp51 + tmp52
    tmp54 = -tmp53
    tmp55 = libdevice.exp(tmp54)
    tmp56 = tl.full([1], 1.0, tl.float32)
    tmp57 = tmp55 + tmp56
    tmp58 = (tmp53 / tmp57)
    tl.store(in_out_ptr0 + (x2), tmp58, xmask)
