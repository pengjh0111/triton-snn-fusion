
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_split_zeros_like_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 20, 'num_store': 16, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 12849152, 'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_split_zeros_like_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (200704 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (401408 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (602112 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (802816 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (1003520 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (1204224 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (1404928 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (1605632 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (1806336 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (2007040 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (2207744 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (2408448 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (2609152 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr0 + (2809856 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr0 + (3010560 + y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp15 - tmp1
    tmp17 = tmp16 * tmp9
    tmp18 = tmp17 * tmp11
    tmp19 = tmp18 + tmp13
    tmp21 = tmp20 - tmp1
    tmp22 = tmp21 * tmp9
    tmp23 = tmp22 * tmp11
    tmp24 = tmp23 + tmp13
    tmp26 = tmp25 - tmp1
    tmp27 = tmp26 * tmp9
    tmp28 = tmp27 * tmp11
    tmp29 = tmp28 + tmp13
    tmp31 = tmp30 - tmp1
    tmp32 = tmp31 * tmp9
    tmp33 = tmp32 * tmp11
    tmp34 = tmp33 + tmp13
    tmp36 = tmp35 - tmp1
    tmp37 = tmp36 * tmp9
    tmp38 = tmp37 * tmp11
    tmp39 = tmp38 + tmp13
    tmp41 = tmp40 - tmp1
    tmp42 = tmp41 * tmp9
    tmp43 = tmp42 * tmp11
    tmp44 = tmp43 + tmp13
    tmp46 = tmp45 - tmp1
    tmp47 = tmp46 * tmp9
    tmp48 = tmp47 * tmp11
    tmp49 = tmp48 + tmp13
    tmp51 = tmp50 - tmp1
    tmp52 = tmp51 * tmp9
    tmp53 = tmp52 * tmp11
    tmp54 = tmp53 + tmp13
    tmp56 = tmp55 - tmp1
    tmp57 = tmp56 * tmp9
    tmp58 = tmp57 * tmp11
    tmp59 = tmp58 + tmp13
    tmp61 = tmp60 - tmp1
    tmp62 = tmp61 * tmp9
    tmp63 = tmp62 * tmp11
    tmp64 = tmp63 + tmp13
    tmp66 = tmp65 - tmp1
    tmp67 = tmp66 * tmp9
    tmp68 = tmp67 * tmp11
    tmp69 = tmp68 + tmp13
    tmp71 = tmp70 - tmp1
    tmp72 = tmp71 * tmp9
    tmp73 = tmp72 * tmp11
    tmp74 = tmp73 + tmp13
    tmp76 = tmp75 - tmp1
    tmp77 = tmp76 * tmp9
    tmp78 = tmp77 * tmp11
    tmp79 = tmp78 + tmp13
    tmp81 = tmp80 - tmp1
    tmp82 = tmp81 * tmp9
    tmp83 = tmp82 * tmp11
    tmp84 = tmp83 + tmp13
    tmp86 = tmp85 - tmp1
    tmp87 = tmp86 * tmp9
    tmp88 = tmp87 * tmp11
    tmp89 = tmp88 + tmp13
    tl.store(out_ptr0 + (x2 + 196*y3), tmp14, xmask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp19, xmask)
    tl.store(out_ptr2 + (x2 + 196*y3), tmp24, xmask)
    tl.store(out_ptr3 + (x2 + 196*y3), tmp29, xmask)
    tl.store(out_ptr4 + (x2 + 196*y3), tmp34, xmask)
    tl.store(out_ptr5 + (x2 + 196*y3), tmp39, xmask)
    tl.store(out_ptr6 + (x2 + 196*y3), tmp44, xmask)
    tl.store(out_ptr7 + (x2 + 196*y3), tmp49, xmask)
    tl.store(out_ptr8 + (x2 + 196*y3), tmp54, xmask)
    tl.store(out_ptr9 + (x2 + 196*y3), tmp59, xmask)
    tl.store(out_ptr10 + (x2 + 196*y3), tmp64, xmask)
    tl.store(out_ptr11 + (x2 + 196*y3), tmp69, xmask)
    tl.store(out_ptr12 + (x2 + 196*y3), tmp74, xmask)
    tl.store(out_ptr13 + (x2 + 196*y3), tmp79, xmask)
    tl.store(out_ptr14 + (x2 + 196*y3), tmp84, xmask)
    tl.store(out_ptr15 + (x2 + 196*y3), tmp89, xmask)
