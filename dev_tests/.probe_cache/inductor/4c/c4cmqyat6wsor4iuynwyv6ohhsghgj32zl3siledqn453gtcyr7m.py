
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_div_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 292000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_div_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1000)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp8 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp14 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x2), xmask)
    tmp20 = tl.load(in_ptr6 + (x2), xmask)
    tmp23 = tl.load(in_ptr7 + (x2), xmask)
    tmp26 = tl.load(in_ptr8 + (x2), xmask)
    tmp29 = tl.load(in_ptr9 + (x2), xmask)
    tmp32 = tl.load(in_ptr10 + (x2), xmask)
    tmp35 = tl.load(in_ptr11 + (x2), xmask)
    tmp38 = tl.load(in_ptr12 + (x2), xmask)
    tmp41 = tl.load(in_ptr13 + (x2), xmask)
    tmp44 = tl.load(in_ptr14 + (x2), xmask)
    tmp47 = tl.load(in_ptr15 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.0, tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp0 + tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp0 + tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp0 + tmp11
    tmp13 = tmp10 + tmp12
    tmp15 = tmp0 + tmp14
    tmp16 = tmp13 + tmp15
    tmp18 = tmp0 + tmp17
    tmp19 = tmp16 + tmp18
    tmp21 = tmp0 + tmp20
    tmp22 = tmp19 + tmp21
    tmp24 = tmp0 + tmp23
    tmp25 = tmp22 + tmp24
    tmp27 = tmp0 + tmp26
    tmp28 = tmp25 + tmp27
    tmp30 = tmp0 + tmp29
    tmp31 = tmp28 + tmp30
    tmp33 = tmp0 + tmp32
    tmp34 = tmp31 + tmp33
    tmp36 = tmp0 + tmp35
    tmp37 = tmp34 + tmp36
    tmp39 = tmp0 + tmp38
    tmp40 = tmp37 + tmp39
    tmp42 = tmp0 + tmp41
    tmp43 = tmp40 + tmp42
    tmp45 = tmp0 + tmp44
    tmp46 = tmp43 + tmp45
    tmp48 = tmp0 + tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = tl.full([1], 0.0625, tl.float32)
    tmp51 = tmp49 * tmp50
    tl.store(in_out_ptr0 + (x2), tmp51, xmask)
