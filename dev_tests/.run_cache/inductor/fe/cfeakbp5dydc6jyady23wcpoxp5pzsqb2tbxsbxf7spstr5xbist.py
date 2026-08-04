
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 79872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    tmp10 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 3, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1], 4, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp5 & xmask, other=0.0)
    tmp9 = tl.where(tmp3, tmp4, tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp12 >= tmp0
    tmp14 = tmp12 < tmp2
    tmp15 = tl.load(in_ptr0 + (1 + 4*x2 + (1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp12 >= tmp2
    tmp17 = tmp12 < tmp6
    tmp18 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp19 = tl.where(tmp14, tmp15, tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 + tmp21
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp23 >= tmp0
    tmp25 = tmp23 < tmp2
    tmp26 = tl.load(in_ptr0 + (1 + 4*x2 + (2)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp23 >= tmp2
    tmp28 = tmp23 < tmp6
    tmp29 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp27 & xmask, other=0.0)
    tmp30 = tl.where(tmp25, tmp26, tmp29)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = tmp2 >= tmp0
    tmp35 = tmp2 < tmp2
    tmp36 = tl.load(in_ptr0 + (1 + 4*x2 + (3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp2 >= tmp2
    tmp38 = tmp2 < tmp6
    tmp39 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp37 & xmask, other=0.0)
    tmp40 = tl.where(tmp35, tmp36, tmp39)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp33 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = -tmp45
    tmp47 = libdevice.exp(tmp46)
    tmp48 = tl.full([1], 1.0, tl.float32)
    tmp49 = tmp47 + tmp48
    tmp50 = (tmp45 / tmp49)
    tl.store(in_out_ptr0 + (x2), tmp50, xmask)
