
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 57, 'num_store': 7, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 448512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_40(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
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
    tmp29 = tl.load(in_ptr4 + (4*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (3 + 4*x2), xmask, eviction_policy='evict_last')
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
    tmp30 = tmp29 * tmp1
    tmp32 = tmp31 * tmp4
    tmp33 = tmp30 + tmp32
    tmp35 = tmp34 * tmp8
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp12
    tmp39 = tmp36 + tmp38
    tmp40 = tmp39 + tmp15
    tmp41 = -tmp16
    tmp42 = libdevice.exp(tmp41)
    tmp43 = tl.full([1], 1.0, tl.float32)
    tmp44 = tmp42 + tmp43
    tmp45 = (tmp16 / tmp44)
    tmp46 = -tmp28
    tmp47 = libdevice.exp(tmp46)
    tmp48 = tmp47 + tmp43
    tmp49 = (tmp28 / tmp48)
    tmp50 = -tmp40
    tmp51 = libdevice.exp(tmp50)
    tmp52 = tmp51 + tmp43
    tmp53 = (tmp40 / tmp52)
    tmp54 = tl.full([1], 0, tl.int64)
    tmp55 = tmp54 >= tmp54
    tmp56 = tl.full([1], 3, tl.int64)
    tmp57 = tmp54 < tmp56
    tmp58 = tl.load(in_ptr0 + (1 + 4*x2 + (0)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp54 >= tmp56
    tmp60 = tl.full([1], 4, tl.int64)
    tmp61 = tmp54 < tmp60
    tmp62 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp59 & xmask, other=0.0)
    tmp63 = tl.where(tmp57, tmp58, tmp62)
    tmp64 = tmp63 * tmp1
    tmp65 = tl.full([1], 1, tl.int64)
    tmp66 = tmp65 >= tmp54
    tmp67 = tmp65 < tmp56
    tmp68 = tl.load(in_ptr0 + (1 + 4*x2 + (1)), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp65 >= tmp56
    tmp70 = tmp65 < tmp60
    tmp71 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp69 & xmask, other=0.0)
    tmp72 = tl.where(tmp67, tmp68, tmp71)
    tmp73 = tmp72 * tmp4
    tmp74 = tmp64 + tmp73
    tmp75 = tl.full([1], 2, tl.int64)
    tmp76 = tmp75 >= tmp54
    tmp77 = tmp75 < tmp56
    tmp78 = tl.load(in_ptr0 + (1 + 4*x2 + (2)), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp75 >= tmp56
    tmp80 = tmp75 < tmp60
    tmp81 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp79 & xmask, other=0.0)
    tmp82 = tl.where(tmp77, tmp78, tmp81)
    tmp83 = tmp82 * tmp8
    tmp84 = tmp74 + tmp83
    tmp85 = tmp56 >= tmp54
    tmp86 = tmp56 < tmp56
    tmp87 = tl.load(in_ptr0 + (1 + 4*x2 + (3)), tmp86 & xmask, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp56 >= tmp56
    tmp89 = tmp56 < tmp60
    tmp90 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp88 & xmask, other=0.0)
    tmp91 = tl.where(tmp86, tmp87, tmp90)
    tmp92 = tmp91 * tmp12
    tmp93 = tmp84 + tmp92
    tmp94 = tl.full([1], 1, tl.int64)
    tmp95 = tl.full([1], 0, tl.int64)
    tmp96 = tmp94 >= tmp95
    tmp97 = tl.full([1], 3, tl.int64)
    tmp98 = tmp94 < tmp97
    tmp99 = tmp98 & tmp57
    tmp100 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (0))), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp94 >= tmp97
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp94 < tmp102
    tmp104 = tmp101 & tmp57
    tmp105 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp106 = tl.where(tmp98, tmp100, tmp105)
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp57, tmp106, tmp107)
    tmp109 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp59 & xmask, other=0.0)
    tmp110 = tl.where(tmp57, tmp108, tmp109)
    tmp111 = tmp110 * tmp1
    tmp112 = tl.full([1], 2, tl.int64)
    tmp113 = tl.full([1], 0, tl.int64)
    tmp114 = tmp112 >= tmp113
    tmp115 = tl.full([1], 3, tl.int64)
    tmp116 = tmp112 < tmp115
    tmp117 = tmp116 & tmp67
    tmp118 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (1))), tmp117 & xmask, eviction_policy='evict_last', other=0.0)
    tmp119 = tmp112 >= tmp115
    tmp120 = tl.full([1], 4, tl.int64)
    tmp121 = tmp112 < tmp120
    tmp122 = tmp119 & tmp67
    tmp123 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp122 & xmask, other=0.0)
    tmp124 = tl.where(tmp116, tmp118, tmp123)
    tmp125 = tl.full(tmp124.shape, 0.0, tmp124.dtype)
    tmp126 = tl.where(tmp67, tmp124, tmp125)
    tmp127 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp69 & xmask, other=0.0)
    tmp128 = tl.where(tmp67, tmp126, tmp127)
    tmp129 = tmp128 * tmp4
    tmp130 = tmp111 + tmp129
    tmp131 = tl.full([1], 3, tl.int64)
    tmp132 = tl.full([1], 0, tl.int64)
    tmp133 = tmp131 >= tmp132
    tmp134 = tmp131 < tmp131
    tmp135 = tmp134 & tmp77
    tmp136 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (2))), tmp135 & xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp131 >= tmp131
    tmp138 = tl.full([1], 4, tl.int64)
    tmp139 = tmp131 < tmp138
    tmp140 = tmp137 & tmp77
    tmp141 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp140 & xmask, other=0.0)
    tmp142 = tl.where(tmp134, tmp136, tmp141)
    tmp143 = tl.full(tmp142.shape, 0.0, tmp142.dtype)
    tmp144 = tl.where(tmp77, tmp142, tmp143)
    tmp145 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp79 & xmask, other=0.0)
    tmp146 = tl.where(tmp77, tmp144, tmp145)
    tmp147 = tmp146 * tmp8
    tmp148 = tmp130 + tmp147
    tmp149 = tl.full([1], 4, tl.int64)
    tmp150 = tl.full([1], 0, tl.int64)
    tmp151 = tmp149 >= tmp150
    tmp152 = tl.full([1], 3, tl.int64)
    tmp153 = tmp149 < tmp152
    tmp154 = tmp153 & tmp86
    tmp155 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (3))), tmp154 & xmask, eviction_policy='evict_last', other=0.0)
    tmp156 = tmp149 >= tmp152
    tmp157 = tmp149 < tmp149
    tmp158 = tmp156 & tmp86
    tmp159 = tl.load(in_ptr5 + (x0 + 3072*x1), tmp158 & xmask, other=0.0)
    tmp160 = tl.where(tmp153, tmp155, tmp159)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp86, tmp160, tmp161)
    tmp163 = tl.load(in_ptr6 + (x0 + 3072*x1), tmp88 & xmask, other=0.0)
    tmp164 = tl.where(tmp86, tmp162, tmp163)
    tmp165 = tmp164 * tmp12
    tmp166 = tmp148 + tmp165
    tmp167 = tmp93 + tmp15
    tmp168 = -tmp167
    tmp169 = libdevice.exp(tmp168)
    tmp170 = tmp169 + tmp43
    tmp171 = (tmp167 / tmp170)
    tmp172 = tmp166 + tmp15
    tmp173 = -tmp172
    tmp174 = libdevice.exp(tmp173)
    tmp175 = tmp174 + tmp43
    tmp176 = (tmp172 / tmp175)
    tmp177 = tl.load(in_ptr3 + (1 + 4*x2 + (0)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp178 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp59 & xmask, other=0.0)
    tmp179 = tl.where(tmp57, tmp177, tmp178)
    tmp180 = tmp179 * tmp1
    tmp181 = tl.load(in_ptr3 + (1 + 4*x2 + (1)), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp182 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp69 & xmask, other=0.0)
    tmp183 = tl.where(tmp67, tmp181, tmp182)
    tmp184 = tmp183 * tmp4
    tmp185 = tmp180 + tmp184
    tmp186 = tl.load(in_ptr3 + (1 + 4*x2 + (2)), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp79 & xmask, other=0.0)
    tmp188 = tl.where(tmp77, tmp186, tmp187)
    tmp189 = tmp188 * tmp8
    tmp190 = tmp185 + tmp189
    tmp191 = tl.load(in_ptr3 + (1 + 4*x2 + (3)), tmp86 & xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp88 & xmask, other=0.0)
    tmp193 = tl.where(tmp86, tmp191, tmp192)
    tmp194 = tmp193 * tmp12
    tmp195 = tmp190 + tmp194
    tmp196 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (0))), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp197 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp198 = tl.where(tmp98, tmp196, tmp197)
    tmp199 = tl.full(tmp198.shape, 0.0, tmp198.dtype)
    tmp200 = tl.where(tmp57, tmp198, tmp199)
    tmp201 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp59 & xmask, other=0.0)
    tmp202 = tl.where(tmp57, tmp200, tmp201)
    tmp203 = tmp202 * tmp1
    tmp204 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (1))), tmp117 & xmask, eviction_policy='evict_last', other=0.0)
    tmp205 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp122 & xmask, other=0.0)
    tmp206 = tl.where(tmp116, tmp204, tmp205)
    tmp207 = tl.full(tmp206.shape, 0.0, tmp206.dtype)
    tmp208 = tl.where(tmp67, tmp206, tmp207)
    tmp209 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp69 & xmask, other=0.0)
    tmp210 = tl.where(tmp67, tmp208, tmp209)
    tmp211 = tmp210 * tmp4
    tmp212 = tmp203 + tmp211
    tmp213 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (2))), tmp135 & xmask, eviction_policy='evict_last', other=0.0)
    tmp214 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp140 & xmask, other=0.0)
    tmp215 = tl.where(tmp134, tmp213, tmp214)
    tmp216 = tl.full(tmp215.shape, 0.0, tmp215.dtype)
    tmp217 = tl.where(tmp77, tmp215, tmp216)
    tmp218 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp79 & xmask, other=0.0)
    tmp219 = tl.where(tmp77, tmp217, tmp218)
    tmp220 = tmp219 * tmp8
    tmp221 = tmp212 + tmp220
    tmp222 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (3))), tmp154 & xmask, eviction_policy='evict_last', other=0.0)
    tmp223 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp158 & xmask, other=0.0)
    tmp224 = tl.where(tmp153, tmp222, tmp223)
    tmp225 = tl.full(tmp224.shape, 0.0, tmp224.dtype)
    tmp226 = tl.where(tmp86, tmp224, tmp225)
    tmp227 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp88 & xmask, other=0.0)
    tmp228 = tl.where(tmp86, tmp226, tmp227)
    tmp229 = tmp228 * tmp12
    tmp230 = tmp221 + tmp229
    tmp231 = tmp195 + tmp15
    tmp232 = -tmp231
    tmp233 = libdevice.exp(tmp232)
    tmp234 = tmp233 + tmp43
    tmp235 = (tmp231 / tmp234)
    tmp236 = tmp230 + tmp15
    tmp237 = -tmp236
    tmp238 = libdevice.exp(tmp237)
    tmp239 = tmp238 + tmp43
    tmp240 = (tmp236 / tmp239)
    tl.store(in_out_ptr0 + (x2), tmp45, xmask)
    tl.store(in_out_ptr1 + (x2), tmp49, xmask)
    tl.store(in_out_ptr2 + (x2), tmp53, xmask)
    tl.store(in_out_ptr3 + (x2), tmp171, xmask)
    tl.store(in_out_ptr4 + (x2), tmp176, xmask)
    tl.store(in_out_ptr5 + (x2), tmp235, xmask)
    tl.store(in_out_ptr6 + (x2), tmp240, xmask)
