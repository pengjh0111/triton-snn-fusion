
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 41, 'num_store': 5, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 325632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_32(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (4*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp17 * tmp1
    tmp20 = tmp19 * tmp4
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp8
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp12
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 + tmp15
    tmp29 = -tmp16
    tmp30 = libdevice.exp(tmp29)
    tmp31 = tl.full([1], 1.0, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = (tmp16 / tmp32)
    tmp34 = -tmp28
    tmp35 = libdevice.exp(tmp34)
    tmp36 = tmp35 + tmp31
    tmp37 = (tmp28 / tmp36)
    tmp38 = tl.full([1], 0, tl.int64)
    tmp39 = tmp38 >= tmp38
    tmp40 = tl.full([1], 3, tl.int64)
    tmp41 = tmp38 < tmp40
    tmp42 = tl.load(in_ptr0 + (1 + 4*x2 + (0)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp38 >= tmp40
    tmp44 = tl.full([1], 4, tl.int64)
    tmp45 = tmp38 < tmp44
    tmp46 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp43 & xmask, other=0.0)
    tmp47 = tl.where(tmp41, tmp42, tmp46)
    tmp48 = tmp47 * tmp1
    tmp49 = tl.full([1], 1, tl.int64)
    tmp50 = tmp49 >= tmp38
    tmp51 = tmp49 < tmp40
    tmp52 = tl.load(in_ptr0 + (1 + 4*x2 + (1)), tmp51 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp49 >= tmp40
    tmp54 = tmp49 < tmp44
    tmp55 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp53 & xmask, other=0.0)
    tmp56 = tl.where(tmp51, tmp52, tmp55)
    tmp57 = tmp56 * tmp4
    tmp58 = tmp48 + tmp57
    tmp59 = tl.full([1], 2, tl.int64)
    tmp60 = tmp59 >= tmp38
    tmp61 = tmp59 < tmp40
    tmp62 = tl.load(in_ptr0 + (1 + 4*x2 + (2)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp59 >= tmp40
    tmp64 = tmp59 < tmp44
    tmp65 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp63 & xmask, other=0.0)
    tmp66 = tl.where(tmp61, tmp62, tmp65)
    tmp67 = tmp66 * tmp8
    tmp68 = tmp58 + tmp67
    tmp69 = tmp40 >= tmp38
    tmp70 = tmp40 < tmp40
    tmp71 = tl.load(in_ptr0 + (1 + 4*x2 + (3)), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp40 >= tmp40
    tmp73 = tmp40 < tmp44
    tmp74 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp72 & xmask, other=0.0)
    tmp75 = tl.where(tmp70, tmp71, tmp74)
    tmp76 = tmp75 * tmp12
    tmp77 = tmp68 + tmp76
    tmp78 = tl.full([1], 1, tl.int64)
    tmp79 = tl.full([1], 0, tl.int64)
    tmp80 = tmp78 >= tmp79
    tmp81 = tl.full([1], 3, tl.int64)
    tmp82 = tmp78 < tmp81
    tmp83 = tmp82 & tmp41
    tmp84 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (0))), tmp83 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp78 >= tmp81
    tmp86 = tl.full([1], 4, tl.int64)
    tmp87 = tmp78 < tmp86
    tmp88 = tmp85 & tmp41
    tmp89 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp88 & xmask, other=0.0)
    tmp90 = tl.where(tmp82, tmp84, tmp89)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp41, tmp90, tmp91)
    tmp93 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp43 & xmask, other=0.0)
    tmp94 = tl.where(tmp41, tmp92, tmp93)
    tmp95 = tmp94 * tmp1
    tmp96 = tl.full([1], 2, tl.int64)
    tmp97 = tl.full([1], 0, tl.int64)
    tmp98 = tmp96 >= tmp97
    tmp99 = tl.full([1], 3, tl.int64)
    tmp100 = tmp96 < tmp99
    tmp101 = tmp100 & tmp51
    tmp102 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (1))), tmp101 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tmp96 >= tmp99
    tmp104 = tl.full([1], 4, tl.int64)
    tmp105 = tmp96 < tmp104
    tmp106 = tmp103 & tmp51
    tmp107 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp106 & xmask, other=0.0)
    tmp108 = tl.where(tmp100, tmp102, tmp107)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp51, tmp108, tmp109)
    tmp111 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp53 & xmask, other=0.0)
    tmp112 = tl.where(tmp51, tmp110, tmp111)
    tmp113 = tmp112 * tmp4
    tmp114 = tmp95 + tmp113
    tmp115 = tl.full([1], 3, tl.int64)
    tmp116 = tl.full([1], 0, tl.int64)
    tmp117 = tmp115 >= tmp116
    tmp118 = tmp115 < tmp115
    tmp119 = tmp118 & tmp61
    tmp120 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (2))), tmp119 & xmask, eviction_policy='evict_last', other=0.0)
    tmp121 = tmp115 >= tmp115
    tmp122 = tl.full([1], 4, tl.int64)
    tmp123 = tmp115 < tmp122
    tmp124 = tmp121 & tmp61
    tmp125 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp126 = tl.where(tmp118, tmp120, tmp125)
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp61, tmp126, tmp127)
    tmp129 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp63 & xmask, other=0.0)
    tmp130 = tl.where(tmp61, tmp128, tmp129)
    tmp131 = tmp130 * tmp8
    tmp132 = tmp114 + tmp131
    tmp133 = tl.full([1], 4, tl.int64)
    tmp134 = tl.full([1], 0, tl.int64)
    tmp135 = tmp133 >= tmp134
    tmp136 = tl.full([1], 3, tl.int64)
    tmp137 = tmp133 < tmp136
    tmp138 = tmp137 & tmp70
    tmp139 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (3))), tmp138 & xmask, eviction_policy='evict_last', other=0.0)
    tmp140 = tmp133 >= tmp136
    tmp141 = tmp133 < tmp133
    tmp142 = tmp140 & tmp70
    tmp143 = tl.load(in_ptr4 + (x0 + 3072*x1), tmp142 & xmask, other=0.0)
    tmp144 = tl.where(tmp137, tmp139, tmp143)
    tmp145 = tl.full(tmp144.shape, 0.0, tmp144.dtype)
    tmp146 = tl.where(tmp70, tmp144, tmp145)
    tmp147 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp72 & xmask, other=0.0)
    tmp148 = tl.where(tmp70, tmp146, tmp147)
    tmp149 = tmp148 * tmp12
    tmp150 = tmp132 + tmp149
    tmp151 = tmp77 + tmp15
    tmp152 = -tmp151
    tmp153 = libdevice.exp(tmp152)
    tmp154 = tmp153 + tmp31
    tmp155 = (tmp151 / tmp154)
    tmp156 = tmp150 + tmp15
    tmp157 = -tmp156
    tmp158 = libdevice.exp(tmp157)
    tmp159 = tmp158 + tmp31
    tmp160 = (tmp156 / tmp159)
    tmp161 = tl.load(in_ptr3 + (1 + 4*x2 + (0)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp43 & xmask, other=0.0)
    tmp163 = tl.where(tmp41, tmp161, tmp162)
    tmp164 = tmp163 * tmp1
    tmp165 = tl.load(in_ptr3 + (1 + 4*x2 + (1)), tmp51 & xmask, eviction_policy='evict_last', other=0.0)
    tmp166 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp53 & xmask, other=0.0)
    tmp167 = tl.where(tmp51, tmp165, tmp166)
    tmp168 = tmp167 * tmp4
    tmp169 = tmp164 + tmp168
    tmp170 = tl.load(in_ptr3 + (1 + 4*x2 + (2)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp171 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp63 & xmask, other=0.0)
    tmp172 = tl.where(tmp61, tmp170, tmp171)
    tmp173 = tmp172 * tmp8
    tmp174 = tmp169 + tmp173
    tmp175 = tl.load(in_ptr3 + (1 + 4*x2 + (3)), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp176 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp72 & xmask, other=0.0)
    tmp177 = tl.where(tmp70, tmp175, tmp176)
    tmp178 = tmp177 * tmp12
    tmp179 = tmp174 + tmp178
    tmp180 = tmp179 + tmp15
    tmp181 = -tmp180
    tmp182 = libdevice.exp(tmp181)
    tmp183 = tmp182 + tmp31
    tmp184 = (tmp180 / tmp183)
    tl.store(in_out_ptr0 + (x2), tmp33, xmask)
    tl.store(in_out_ptr1 + (x2), tmp37, xmask)
    tl.store(in_out_ptr2 + (x2), tmp155, xmask)
    tl.store(in_out_ptr3 + (x2), tmp160, xmask)
    tl.store(in_out_ptr4 + (x2), tmp184, xmask)
