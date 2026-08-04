
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 104448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    tmp24 = tl.load(in_ptr3 + (4*x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr3 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr3 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp10 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (0))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp4 >= tmp7
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp4 < tmp12
    tmp14 = tmp11 & tmp3
    tmp15 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp8, tmp10, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp3, tmp16, tmp17)
    tmp19 = tmp0 >= tmp2
    tmp20 = tl.full([1], 4, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp19 & xmask, other=0.0)
    tmp23 = tl.where(tmp3, tmp18, tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp26 >= tmp0
    tmp28 = tmp26 < tmp2
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tl.full([1], 0, tl.int64)
    tmp31 = tmp29 >= tmp30
    tmp32 = tl.full([1], 3, tl.int64)
    tmp33 = tmp29 < tmp32
    tmp34 = tmp33 & tmp28
    tmp35 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (1))), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp29 >= tmp32
    tmp37 = tl.full([1], 4, tl.int64)
    tmp38 = tmp29 < tmp37
    tmp39 = tmp36 & tmp28
    tmp40 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp33, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp28, tmp41, tmp42)
    tmp44 = tmp26 >= tmp2
    tmp45 = tmp26 < tmp20
    tmp46 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp44 & xmask, other=0.0)
    tmp47 = tl.where(tmp28, tmp43, tmp46)
    tmp49 = tmp47 * tmp48
    tmp50 = tmp25 + tmp49
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp51 >= tmp0
    tmp53 = tmp51 < tmp2
    tmp54 = tl.full([1], 3, tl.int64)
    tmp55 = tl.full([1], 0, tl.int64)
    tmp56 = tmp54 >= tmp55
    tmp57 = tmp54 < tmp54
    tmp58 = tmp57 & tmp53
    tmp59 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (2))), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp54 >= tmp54
    tmp61 = tl.full([1], 4, tl.int64)
    tmp62 = tmp54 < tmp61
    tmp63 = tmp60 & tmp53
    tmp64 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp63 & xmask, other=0.0)
    tmp65 = tl.where(tmp57, tmp59, tmp64)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp53, tmp65, tmp66)
    tmp68 = tmp51 >= tmp2
    tmp69 = tmp51 < tmp20
    tmp70 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp68 & xmask, other=0.0)
    tmp71 = tl.where(tmp53, tmp67, tmp70)
    tmp73 = tmp71 * tmp72
    tmp74 = tmp50 + tmp73
    tmp75 = tmp2 >= tmp0
    tmp76 = tmp2 < tmp2
    tmp77 = tl.full([1], 4, tl.int64)
    tmp78 = tl.full([1], 0, tl.int64)
    tmp79 = tmp77 >= tmp78
    tmp80 = tl.full([1], 3, tl.int64)
    tmp81 = tmp77 < tmp80
    tmp82 = tmp81 & tmp76
    tmp83 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (3))), tmp82 & xmask, eviction_policy='evict_last', other=0.0)
    tmp84 = tmp77 >= tmp80
    tmp85 = tmp77 < tmp77
    tmp86 = tmp84 & tmp76
    tmp87 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp86 & xmask, other=0.0)
    tmp88 = tl.where(tmp81, tmp83, tmp87)
    tmp89 = tl.full(tmp88.shape, 0.0, tmp88.dtype)
    tmp90 = tl.where(tmp76, tmp88, tmp89)
    tmp91 = tmp2 >= tmp2
    tmp92 = tmp2 < tmp20
    tmp93 = tl.load(in_ptr2 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp94 = tl.where(tmp76, tmp90, tmp93)
    tmp96 = tmp94 * tmp95
    tmp97 = tmp74 + tmp96
    tmp99 = tmp97 + tmp98
    tmp100 = -tmp99
    tmp101 = libdevice.exp(tmp100)
    tmp102 = tl.full([1], 1.0, tl.float32)
    tmp103 = tmp101 + tmp102
    tmp104 = (tmp99 / tmp103)
    tl.store(in_out_ptr0 + (x2), tmp104, xmask)
