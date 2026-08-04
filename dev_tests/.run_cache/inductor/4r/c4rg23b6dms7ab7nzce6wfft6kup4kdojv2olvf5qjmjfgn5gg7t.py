
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_slice_split_unsqueeze_zeros_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_slice_split_unsqueeze_zeros_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 1536)
    x2 = xindex // 6144
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (x0)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 3, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = 1 + (1 + (x0))
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 3, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.full([1], 0.0, tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tmp11 >= tmp14
    tmp21 = tl.full([1], 4, tl.int64)
    tmp22 = tmp11 < tmp21
    tmp23 = tmp20 & tmp10
    tmp24 = tl.load(in_ptr0 + (x1 + 3072*x2), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp15, tmp19, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tmp5 >= tmp8
    tmp29 = tl.full([1], 4, tl.int64)
    tmp30 = tmp5 < tmp29
    tmp31 = tmp28 & tmp4
    tmp32 = tl.load(in_ptr1 + (x1 + 3072*x2), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp9, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp4, tmp33, tmp34)
    tmp36 = tmp0 >= tmp3
    tmp37 = tl.full([1], 4, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr2 + (x1 + 3072*x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp4, tmp35, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, None)
