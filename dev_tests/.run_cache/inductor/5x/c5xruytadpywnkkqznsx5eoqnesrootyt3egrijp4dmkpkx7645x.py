
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 61, 'num_store': 7, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 473088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_41(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp117 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr5 + (4*x2), xmask, eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr5 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr5 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp137 = tl.load(in_ptr5 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr6 + (4*x2), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr6 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr6 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp149 = tl.load(in_ptr6 + (3 + 4*x2), xmask, eviction_policy='evict_last')
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
    tmp44 = tl.full([1], 1, tl.int64)
    tmp45 = tl.full([1], 0, tl.int64)
    tmp46 = tmp44 >= tmp45
    tmp47 = tl.full([1], 3, tl.int64)
    tmp48 = tmp44 < tmp47
    tmp49 = tmp48 & tmp3
    tmp50 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (0))), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp44 >= tmp47
    tmp52 = tl.full([1], 4, tl.int64)
    tmp53 = tmp44 < tmp52
    tmp54 = tmp51 & tmp3
    tmp55 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp54 & xmask, other=0.0)
    tmp56 = tl.where(tmp48, tmp50, tmp55)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp3, tmp56, tmp57)
    tmp59 = tl.load(in_ptr3 + (x0 + 3072*x1), tmp5 & xmask, other=0.0)
    tmp60 = tl.where(tmp3, tmp58, tmp59)
    tmp61 = tmp60 * tmp10
    tmp62 = tl.full([1], 2, tl.int64)
    tmp63 = tl.full([1], 0, tl.int64)
    tmp64 = tmp62 >= tmp63
    tmp65 = tl.full([1], 3, tl.int64)
    tmp66 = tmp62 < tmp65
    tmp67 = tmp66 & tmp14
    tmp68 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (1))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp62 >= tmp65
    tmp70 = tl.full([1], 4, tl.int64)
    tmp71 = tmp62 < tmp70
    tmp72 = tmp69 & tmp14
    tmp73 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp72 & xmask, other=0.0)
    tmp74 = tl.where(tmp66, tmp68, tmp73)
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp14, tmp74, tmp75)
    tmp77 = tl.load(in_ptr3 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp78 = tl.where(tmp14, tmp76, tmp77)
    tmp79 = tmp78 * tmp20
    tmp80 = tmp61 + tmp79
    tmp81 = tl.full([1], 3, tl.int64)
    tmp82 = tl.full([1], 0, tl.int64)
    tmp83 = tmp81 >= tmp82
    tmp84 = tmp81 < tmp81
    tmp85 = tmp84 & tmp25
    tmp86 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (2))), tmp85 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp81 >= tmp81
    tmp88 = tl.full([1], 4, tl.int64)
    tmp89 = tmp81 < tmp88
    tmp90 = tmp87 & tmp25
    tmp91 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp90 & xmask, other=0.0)
    tmp92 = tl.where(tmp84, tmp86, tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp25, tmp92, tmp93)
    tmp95 = tl.load(in_ptr3 + (x0 + 3072*x1), tmp27 & xmask, other=0.0)
    tmp96 = tl.where(tmp25, tmp94, tmp95)
    tmp97 = tmp96 * tmp31
    tmp98 = tmp80 + tmp97
    tmp99 = tl.full([1], 4, tl.int64)
    tmp100 = tl.full([1], 0, tl.int64)
    tmp101 = tmp99 >= tmp100
    tmp102 = tl.full([1], 3, tl.int64)
    tmp103 = tmp99 < tmp102
    tmp104 = tmp103 & tmp35
    tmp105 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (3))), tmp104 & xmask, eviction_policy='evict_last', other=0.0)
    tmp106 = tmp99 >= tmp102
    tmp107 = tmp99 < tmp99
    tmp108 = tmp106 & tmp35
    tmp109 = tl.load(in_ptr1 + (x0 + 3072*x1), tmp108 & xmask, other=0.0)
    tmp110 = tl.where(tmp103, tmp105, tmp109)
    tmp111 = tl.full(tmp110.shape, 0.0, tmp110.dtype)
    tmp112 = tl.where(tmp35, tmp110, tmp111)
    tmp113 = tl.load(in_ptr3 + (x0 + 3072*x1), tmp37 & xmask, other=0.0)
    tmp114 = tl.where(tmp35, tmp112, tmp113)
    tmp115 = tmp114 * tmp41
    tmp116 = tmp98 + tmp115
    tmp118 = tmp43 + tmp117
    tmp119 = -tmp118
    tmp120 = libdevice.exp(tmp119)
    tmp121 = tl.full([1], 1.0, tl.float32)
    tmp122 = tmp120 + tmp121
    tmp123 = (tmp118 / tmp122)
    tmp124 = tmp116 + tmp117
    tmp125 = -tmp124
    tmp126 = libdevice.exp(tmp125)
    tmp127 = tmp126 + tmp121
    tmp128 = (tmp124 / tmp127)
    tmp130 = tmp129 * tmp10
    tmp132 = tmp131 * tmp20
    tmp133 = tmp130 + tmp132
    tmp135 = tmp134 * tmp31
    tmp136 = tmp133 + tmp135
    tmp138 = tmp137 * tmp41
    tmp139 = tmp136 + tmp138
    tmp140 = tmp139 + tmp117
    tmp142 = tmp141 * tmp10
    tmp144 = tmp143 * tmp20
    tmp145 = tmp142 + tmp144
    tmp147 = tmp146 * tmp31
    tmp148 = tmp145 + tmp147
    tmp150 = tmp149 * tmp41
    tmp151 = tmp148 + tmp150
    tmp152 = tmp151 + tmp117
    tmp153 = -tmp140
    tmp154 = libdevice.exp(tmp153)
    tmp155 = tmp154 + tmp121
    tmp156 = (tmp140 / tmp155)
    tmp157 = -tmp152
    tmp158 = libdevice.exp(tmp157)
    tmp159 = tmp158 + tmp121
    tmp160 = (tmp152 / tmp159)
    tmp161 = tl.load(in_ptr5 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp5 & xmask, other=0.0)
    tmp163 = tl.where(tmp3, tmp161, tmp162)
    tmp164 = tmp163 * tmp10
    tmp165 = tl.load(in_ptr5 + (1 + 4*x2 + (1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp166 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp167 = tl.where(tmp14, tmp165, tmp166)
    tmp168 = tmp167 * tmp20
    tmp169 = tmp164 + tmp168
    tmp170 = tl.load(in_ptr5 + (1 + 4*x2 + (2)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp171 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp27 & xmask, other=0.0)
    tmp172 = tl.where(tmp25, tmp170, tmp171)
    tmp173 = tmp172 * tmp31
    tmp174 = tmp169 + tmp173
    tmp175 = tl.load(in_ptr5 + (1 + 4*x2 + (3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp176 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp37 & xmask, other=0.0)
    tmp177 = tl.where(tmp35, tmp175, tmp176)
    tmp178 = tmp177 * tmp41
    tmp179 = tmp174 + tmp178
    tmp180 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (0))), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp181 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp54 & xmask, other=0.0)
    tmp182 = tl.where(tmp48, tmp180, tmp181)
    tmp183 = tl.full(tmp182.shape, 0.0, tmp182.dtype)
    tmp184 = tl.where(tmp3, tmp182, tmp183)
    tmp185 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp5 & xmask, other=0.0)
    tmp186 = tl.where(tmp3, tmp184, tmp185)
    tmp187 = tmp186 * tmp10
    tmp188 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (1))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp189 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp72 & xmask, other=0.0)
    tmp190 = tl.where(tmp66, tmp188, tmp189)
    tmp191 = tl.full(tmp190.shape, 0.0, tmp190.dtype)
    tmp192 = tl.where(tmp14, tmp190, tmp191)
    tmp193 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp194 = tl.where(tmp14, tmp192, tmp193)
    tmp195 = tmp194 * tmp20
    tmp196 = tmp187 + tmp195
    tmp197 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (2))), tmp85 & xmask, eviction_policy='evict_last', other=0.0)
    tmp198 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp90 & xmask, other=0.0)
    tmp199 = tl.where(tmp84, tmp197, tmp198)
    tmp200 = tl.full(tmp199.shape, 0.0, tmp199.dtype)
    tmp201 = tl.where(tmp25, tmp199, tmp200)
    tmp202 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp27 & xmask, other=0.0)
    tmp203 = tl.where(tmp25, tmp201, tmp202)
    tmp204 = tmp203 * tmp31
    tmp205 = tmp196 + tmp204
    tmp206 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (3))), tmp104 & xmask, eviction_policy='evict_last', other=0.0)
    tmp207 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp108 & xmask, other=0.0)
    tmp208 = tl.where(tmp103, tmp206, tmp207)
    tmp209 = tl.full(tmp208.shape, 0.0, tmp208.dtype)
    tmp210 = tl.where(tmp35, tmp208, tmp209)
    tmp211 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp37 & xmask, other=0.0)
    tmp212 = tl.where(tmp35, tmp210, tmp211)
    tmp213 = tmp212 * tmp41
    tmp214 = tmp205 + tmp213
    tmp215 = tmp179 + tmp117
    tmp216 = -tmp215
    tmp217 = libdevice.exp(tmp216)
    tmp218 = tmp217 + tmp121
    tmp219 = (tmp215 / tmp218)
    tmp220 = tmp214 + tmp117
    tmp221 = -tmp220
    tmp222 = libdevice.exp(tmp221)
    tmp223 = tmp222 + tmp121
    tmp224 = (tmp220 / tmp223)
    tmp225 = tl.load(in_ptr6 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp226 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp5 & xmask, other=0.0)
    tmp227 = tl.where(tmp3, tmp225, tmp226)
    tmp228 = tmp227 * tmp10
    tmp229 = tl.load(in_ptr6 + (1 + 4*x2 + (1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp230 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp16 & xmask, other=0.0)
    tmp231 = tl.where(tmp14, tmp229, tmp230)
    tmp232 = tmp231 * tmp20
    tmp233 = tmp228 + tmp232
    tmp234 = tl.load(in_ptr6 + (1 + 4*x2 + (2)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp235 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp27 & xmask, other=0.0)
    tmp236 = tl.where(tmp25, tmp234, tmp235)
    tmp237 = tmp236 * tmp31
    tmp238 = tmp233 + tmp237
    tmp239 = tl.load(in_ptr6 + (1 + 4*x2 + (3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp240 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp37 & xmask, other=0.0)
    tmp241 = tl.where(tmp35, tmp239, tmp240)
    tmp242 = tmp241 * tmp41
    tmp243 = tmp238 + tmp242
    tmp244 = tmp243 + tmp117
    tmp245 = -tmp244
    tmp246 = libdevice.exp(tmp245)
    tmp247 = tmp246 + tmp121
    tmp248 = (tmp244 / tmp247)
    tl.store(in_out_ptr0 + (x2), tmp123, xmask)
    tl.store(in_out_ptr1 + (x2), tmp128, xmask)
    tl.store(in_out_ptr2 + (x2), tmp156, xmask)
    tl.store(in_out_ptr3 + (x2), tmp160, xmask)
    tl.store(in_out_ptr4 + (x2), tmp219, xmask)
    tl.store(in_out_ptr5 + (x2), tmp224, xmask)
    tl.store(in_out_ptr6 + (x2), tmp248, xmask)
