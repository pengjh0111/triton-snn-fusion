
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_lif_forward_state_38', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 22, 'num_store': 7, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 11247616, 'x': 11239424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_lif_forward_state_38(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr1 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_out_ptr2 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr10 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_out_ptr3 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr11 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_out_ptr4 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr12 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_out_ptr5 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr13 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp84 = tl.load(in_out_ptr6 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr14 + (y0 + 256*x2 + 50176*y1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt_rn(tmp19)
    tmp21 = (tmp7 / tmp20)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp30 = tmp29 - tmp1
    tmp31 = tmp30 * tmp9
    tmp32 = tmp31 * tmp11
    tmp33 = tmp32 + tmp13
    tmp35 = tmp34 - tmp16
    tmp36 = tmp35 * tmp22
    tmp37 = tmp36 * tmp24
    tmp38 = tmp37 + tmp26
    tmp39 = tmp33 + tmp38
    tmp41 = tmp40 - tmp1
    tmp42 = tmp41 * tmp9
    tmp43 = tmp42 * tmp11
    tmp44 = tmp43 + tmp13
    tmp46 = tmp45 - tmp16
    tmp47 = tmp46 * tmp22
    tmp48 = tmp47 * tmp24
    tmp49 = tmp48 + tmp26
    tmp50 = tmp44 + tmp49
    tmp52 = tmp51 - tmp1
    tmp53 = tmp52 * tmp9
    tmp54 = tmp53 * tmp11
    tmp55 = tmp54 + tmp13
    tmp57 = tmp56 - tmp16
    tmp58 = tmp57 * tmp22
    tmp59 = tmp58 * tmp24
    tmp60 = tmp59 + tmp26
    tmp61 = tmp55 + tmp60
    tmp63 = tmp62 - tmp1
    tmp64 = tmp63 * tmp9
    tmp65 = tmp64 * tmp11
    tmp66 = tmp65 + tmp13
    tmp68 = tmp67 - tmp16
    tmp69 = tmp68 * tmp22
    tmp70 = tmp69 * tmp24
    tmp71 = tmp70 + tmp26
    tmp72 = tmp66 + tmp71
    tmp74 = tmp73 - tmp1
    tmp75 = tmp74 * tmp9
    tmp76 = tmp75 * tmp11
    tmp77 = tmp76 + tmp13
    tmp79 = tmp78 - tmp16
    tmp80 = tmp79 * tmp22
    tmp81 = tmp80 * tmp24
    tmp82 = tmp81 + tmp26
    tmp83 = tmp77 + tmp82
    tmp85 = tmp84 - tmp1
    tmp86 = tmp85 * tmp9
    tmp87 = tmp86 * tmp11
    tmp88 = tmp87 + tmp13
    tmp90 = tmp89 - tmp16
    tmp91 = tmp90 * tmp22
    tmp92 = tmp91 * tmp24
    tmp93 = tmp92 + tmp26
    tmp94 = tmp88 + tmp93
    tl.store(out_ptr0 + (x2 + 196*y3), tmp61, xmask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp50, xmask)
    tl.store(out_ptr2 + (x2 + 196*y3), tmp72, xmask)
    tl.store(out_ptr3 + (x2 + 196*y3), tmp39, xmask)
    tl.store(out_ptr4 + (x2 + 196*y3), tmp83, xmask)
    tl.store(out_ptr5 + (x2 + 196*y3), tmp28, xmask)
    tl.store(out_ptr6 + (x2 + 196*y3), tmp94, xmask)
