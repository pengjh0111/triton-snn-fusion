
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_out_ptr9': '*fp32', 'in_out_ptr10': '*fp32', 'in_out_ptr11': '*fp32', 'in_out_ptr12': '*fp32', 'in_out_ptr13': '*fp32', 'in_out_ptr14': '*fp32', 'in_out_ptr15': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr10', 'in_out_ptr11', 'in_out_ptr12', 'in_out_ptr13', 'in_out_ptr14', 'in_out_ptr15', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 125, 'num_store': 16, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1062912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_zeros_22(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_out_ptr10, in_out_ptr11, in_out_ptr12, in_out_ptr13, in_out_ptr14, in_out_ptr15, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
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
    tmp133 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr3 + (4*x2), xmask, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr3 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr3 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr3 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr4 + (4*x2), xmask, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr4 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr4 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr4 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr5 + (4*x2), xmask, eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr5 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr5 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr5 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp181 = tl.load(in_ptr6 + (4*x2), xmask, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr6 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr6 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr6 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp193 = tl.load(in_ptr7 + (4*x2), xmask, eviction_policy='evict_last')
    tmp195 = tl.load(in_ptr7 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr7 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp201 = tl.load(in_ptr7 + (3 + 4*x2), xmask, eviction_policy='evict_last')
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
    tmp52 = tl.full([1], 1, tl.int64)
    tmp53 = tl.full([1], 0, tl.int64)
    tmp54 = tmp52 >= tmp53
    tmp55 = tl.full([1], 3, tl.int64)
    tmp56 = tmp52 < tmp55
    tmp57 = tmp56 & tmp3
    tmp58 = tl.full([1], 0.0, tl.float32)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp57, tmp58, tmp59)
    tmp61 = tmp52 >= tmp55
    tmp62 = tl.full([1], 4, tl.int64)
    tmp63 = tmp52 < tmp62
    tmp64 = tmp61 & tmp3
    tmp65 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp66 = tl.where(tmp56, tmp60, tmp65)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp3, tmp66, tmp67)
    tmp69 = tl.load(in_ptr0 + (12288 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp70 = tl.where(tmp3, tmp68, tmp69)
    tmp71 = tmp70 * tmp12
    tmp72 = tl.full([1], 2, tl.int64)
    tmp73 = tl.full([1], 0, tl.int64)
    tmp74 = tmp72 >= tmp73
    tmp75 = tl.full([1], 3, tl.int64)
    tmp76 = tmp72 < tmp75
    tmp77 = tmp76 & tmp16
    tmp78 = tl.full([1], 0.0, tl.float32)
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp77, tmp78, tmp79)
    tmp81 = tmp72 >= tmp75
    tmp82 = tl.full([1], 4, tl.int64)
    tmp83 = tmp72 < tmp82
    tmp84 = tmp81 & tmp16
    tmp85 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp86 = tl.where(tmp76, tmp80, tmp85)
    tmp87 = tl.full(tmp86.shape, 0.0, tmp86.dtype)
    tmp88 = tl.where(tmp16, tmp86, tmp87)
    tmp89 = tl.load(in_ptr0 + (12288 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp90 = tl.where(tmp16, tmp88, tmp89)
    tmp91 = tmp90 * tmp24
    tmp92 = tmp71 + tmp91
    tmp93 = tl.full([1], 3, tl.int64)
    tmp94 = tl.full([1], 0, tl.int64)
    tmp95 = tmp93 >= tmp94
    tmp96 = tmp93 < tmp93
    tmp97 = tmp96 & tmp29
    tmp98 = tl.full([1], 0.0, tl.float32)
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp97, tmp98, tmp99)
    tmp101 = tmp93 >= tmp93
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp93 < tmp102
    tmp104 = tmp101 & tmp29
    tmp105 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp106 = tl.where(tmp96, tmp100, tmp105)
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp29, tmp106, tmp107)
    tmp109 = tl.load(in_ptr0 + (12288 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp110 = tl.where(tmp29, tmp108, tmp109)
    tmp111 = tmp110 * tmp37
    tmp112 = tmp92 + tmp111
    tmp113 = tl.full([1], 4, tl.int64)
    tmp114 = tl.full([1], 0, tl.int64)
    tmp115 = tmp113 >= tmp114
    tmp116 = tl.full([1], 3, tl.int64)
    tmp117 = tmp113 < tmp116
    tmp118 = tmp117 & tmp41
    tmp119 = tl.full([1], 0.0, tl.float32)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp118, tmp119, tmp120)
    tmp122 = tmp113 >= tmp116
    tmp123 = tmp113 < tmp113
    tmp124 = tmp122 & tmp41
    tmp125 = tl.load(in_ptr0 + (x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp126 = tl.where(tmp117, tmp121, tmp125)
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp41, tmp126, tmp127)
    tmp129 = tl.load(in_ptr0 + (12288 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp130 = tl.where(tmp41, tmp128, tmp129)
    tmp131 = tmp130 * tmp49
    tmp132 = tmp112 + tmp131
    tmp134 = tmp51 + tmp133
    tmp135 = -tmp134
    tmp136 = libdevice.exp(tmp135)
    tmp137 = tl.full([1], 1.0, tl.float32)
    tmp138 = tmp136 + tmp137
    tmp139 = (tmp134 / tmp138)
    tmp140 = tmp132 + tmp133
    tmp141 = -tmp140
    tmp142 = libdevice.exp(tmp141)
    tmp143 = tmp142 + tmp137
    tmp144 = (tmp140 / tmp143)
    tmp146 = tmp145 * tmp12
    tmp148 = tmp147 * tmp24
    tmp149 = tmp146 + tmp148
    tmp151 = tmp150 * tmp37
    tmp152 = tmp149 + tmp151
    tmp154 = tmp153 * tmp49
    tmp155 = tmp152 + tmp154
    tmp156 = tmp155 + tmp133
    tmp158 = tmp157 * tmp12
    tmp160 = tmp159 * tmp24
    tmp161 = tmp158 + tmp160
    tmp163 = tmp162 * tmp37
    tmp164 = tmp161 + tmp163
    tmp166 = tmp165 * tmp49
    tmp167 = tmp164 + tmp166
    tmp168 = tmp167 + tmp133
    tmp170 = tmp169 * tmp12
    tmp172 = tmp171 * tmp24
    tmp173 = tmp170 + tmp172
    tmp175 = tmp174 * tmp37
    tmp176 = tmp173 + tmp175
    tmp178 = tmp177 * tmp49
    tmp179 = tmp176 + tmp178
    tmp180 = tmp179 + tmp133
    tmp182 = tmp181 * tmp12
    tmp184 = tmp183 * tmp24
    tmp185 = tmp182 + tmp184
    tmp187 = tmp186 * tmp37
    tmp188 = tmp185 + tmp187
    tmp190 = tmp189 * tmp49
    tmp191 = tmp188 + tmp190
    tmp192 = tmp191 + tmp133
    tmp194 = tmp193 * tmp12
    tmp196 = tmp195 * tmp24
    tmp197 = tmp194 + tmp196
    tmp199 = tmp198 * tmp37
    tmp200 = tmp197 + tmp199
    tmp202 = tmp201 * tmp49
    tmp203 = tmp200 + tmp202
    tmp204 = tmp203 + tmp133
    tmp205 = -tmp156
    tmp206 = libdevice.exp(tmp205)
    tmp207 = tmp206 + tmp137
    tmp208 = (tmp156 / tmp207)
    tmp209 = -tmp168
    tmp210 = libdevice.exp(tmp209)
    tmp211 = tmp210 + tmp137
    tmp212 = (tmp168 / tmp211)
    tmp213 = -tmp180
    tmp214 = libdevice.exp(tmp213)
    tmp215 = tmp214 + tmp137
    tmp216 = (tmp180 / tmp215)
    tmp217 = -tmp192
    tmp218 = libdevice.exp(tmp217)
    tmp219 = tmp218 + tmp137
    tmp220 = (tmp192 / tmp219)
    tmp221 = -tmp204
    tmp222 = libdevice.exp(tmp221)
    tmp223 = tmp222 + tmp137
    tmp224 = (tmp204 / tmp223)
    tmp225 = tl.load(in_ptr3 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp226 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp227 = tl.where(tmp3, tmp225, tmp226)
    tmp228 = tmp227 * tmp12
    tmp229 = tl.load(in_ptr3 + (1 + 4*x2 + (1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp230 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp231 = tl.where(tmp16, tmp229, tmp230)
    tmp232 = tmp231 * tmp24
    tmp233 = tmp228 + tmp232
    tmp234 = tl.load(in_ptr3 + (1 + 4*x2 + (2)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp235 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp236 = tl.where(tmp29, tmp234, tmp235)
    tmp237 = tmp236 * tmp37
    tmp238 = tmp233 + tmp237
    tmp239 = tl.load(in_ptr3 + (1 + 4*x2 + (3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp240 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp241 = tl.where(tmp41, tmp239, tmp240)
    tmp242 = tmp241 * tmp49
    tmp243 = tmp238 + tmp242
    tmp244 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (0))), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp245 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp246 = tl.where(tmp56, tmp244, tmp245)
    tmp247 = tl.full(tmp246.shape, 0.0, tmp246.dtype)
    tmp248 = tl.where(tmp3, tmp246, tmp247)
    tmp249 = tl.load(in_ptr0 + (49152 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp250 = tl.where(tmp3, tmp248, tmp249)
    tmp251 = tmp250 * tmp12
    tmp252 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (1))), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp253 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp254 = tl.where(tmp76, tmp252, tmp253)
    tmp255 = tl.full(tmp254.shape, 0.0, tmp254.dtype)
    tmp256 = tl.where(tmp16, tmp254, tmp255)
    tmp257 = tl.load(in_ptr0 + (49152 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp258 = tl.where(tmp16, tmp256, tmp257)
    tmp259 = tmp258 * tmp24
    tmp260 = tmp251 + tmp259
    tmp261 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (2))), tmp97 & xmask, eviction_policy='evict_last', other=0.0)
    tmp262 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp263 = tl.where(tmp96, tmp261, tmp262)
    tmp264 = tl.full(tmp263.shape, 0.0, tmp263.dtype)
    tmp265 = tl.where(tmp29, tmp263, tmp264)
    tmp266 = tl.load(in_ptr0 + (49152 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp267 = tl.where(tmp29, tmp265, tmp266)
    tmp268 = tmp267 * tmp37
    tmp269 = tmp260 + tmp268
    tmp270 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (3))), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp271 = tl.load(in_ptr0 + (36864 + x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp272 = tl.where(tmp117, tmp270, tmp271)
    tmp273 = tl.full(tmp272.shape, 0.0, tmp272.dtype)
    tmp274 = tl.where(tmp41, tmp272, tmp273)
    tmp275 = tl.load(in_ptr0 + (49152 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp276 = tl.where(tmp41, tmp274, tmp275)
    tmp277 = tmp276 * tmp49
    tmp278 = tmp269 + tmp277
    tmp279 = tmp243 + tmp133
    tmp280 = -tmp279
    tmp281 = libdevice.exp(tmp280)
    tmp282 = tmp281 + tmp137
    tmp283 = (tmp279 / tmp282)
    tmp284 = tmp278 + tmp133
    tmp285 = -tmp284
    tmp286 = libdevice.exp(tmp285)
    tmp287 = tmp286 + tmp137
    tmp288 = (tmp284 / tmp287)
    tmp289 = tl.load(in_ptr4 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp290 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp291 = tl.where(tmp3, tmp289, tmp290)
    tmp292 = tmp291 * tmp12
    tmp293 = tl.load(in_ptr4 + (1 + 4*x2 + (1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp294 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp295 = tl.where(tmp16, tmp293, tmp294)
    tmp296 = tmp295 * tmp24
    tmp297 = tmp292 + tmp296
    tmp298 = tl.load(in_ptr4 + (1 + 4*x2 + (2)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp299 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp300 = tl.where(tmp29, tmp298, tmp299)
    tmp301 = tmp300 * tmp37
    tmp302 = tmp297 + tmp301
    tmp303 = tl.load(in_ptr4 + (1 + 4*x2 + (3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp304 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp305 = tl.where(tmp41, tmp303, tmp304)
    tmp306 = tmp305 * tmp49
    tmp307 = tmp302 + tmp306
    tmp308 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (0))), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp309 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp310 = tl.where(tmp56, tmp308, tmp309)
    tmp311 = tl.full(tmp310.shape, 0.0, tmp310.dtype)
    tmp312 = tl.where(tmp3, tmp310, tmp311)
    tmp313 = tl.load(in_ptr0 + (86016 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp314 = tl.where(tmp3, tmp312, tmp313)
    tmp315 = tmp314 * tmp12
    tmp316 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (1))), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp317 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp318 = tl.where(tmp76, tmp316, tmp317)
    tmp319 = tl.full(tmp318.shape, 0.0, tmp318.dtype)
    tmp320 = tl.where(tmp16, tmp318, tmp319)
    tmp321 = tl.load(in_ptr0 + (86016 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp322 = tl.where(tmp16, tmp320, tmp321)
    tmp323 = tmp322 * tmp24
    tmp324 = tmp315 + tmp323
    tmp325 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (2))), tmp97 & xmask, eviction_policy='evict_last', other=0.0)
    tmp326 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp327 = tl.where(tmp96, tmp325, tmp326)
    tmp328 = tl.full(tmp327.shape, 0.0, tmp327.dtype)
    tmp329 = tl.where(tmp29, tmp327, tmp328)
    tmp330 = tl.load(in_ptr0 + (86016 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp331 = tl.where(tmp29, tmp329, tmp330)
    tmp332 = tmp331 * tmp37
    tmp333 = tmp324 + tmp332
    tmp334 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (3))), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp335 = tl.load(in_ptr0 + (73728 + x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp336 = tl.where(tmp117, tmp334, tmp335)
    tmp337 = tl.full(tmp336.shape, 0.0, tmp336.dtype)
    tmp338 = tl.where(tmp41, tmp336, tmp337)
    tmp339 = tl.load(in_ptr0 + (86016 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp340 = tl.where(tmp41, tmp338, tmp339)
    tmp341 = tmp340 * tmp49
    tmp342 = tmp333 + tmp341
    tmp343 = tmp307 + tmp133
    tmp344 = -tmp343
    tmp345 = libdevice.exp(tmp344)
    tmp346 = tmp345 + tmp137
    tmp347 = (tmp343 / tmp346)
    tmp348 = tmp342 + tmp133
    tmp349 = -tmp348
    tmp350 = libdevice.exp(tmp349)
    tmp351 = tmp350 + tmp137
    tmp352 = (tmp348 / tmp351)
    tmp353 = tl.load(in_ptr5 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp354 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp355 = tl.where(tmp3, tmp353, tmp354)
    tmp356 = tmp355 * tmp12
    tmp357 = tl.load(in_ptr5 + (1 + 4*x2 + (1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp358 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp359 = tl.where(tmp16, tmp357, tmp358)
    tmp360 = tmp359 * tmp24
    tmp361 = tmp356 + tmp360
    tmp362 = tl.load(in_ptr5 + (1 + 4*x2 + (2)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp363 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp364 = tl.where(tmp29, tmp362, tmp363)
    tmp365 = tmp364 * tmp37
    tmp366 = tmp361 + tmp365
    tmp367 = tl.load(in_ptr5 + (1 + 4*x2 + (3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp368 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp369 = tl.where(tmp41, tmp367, tmp368)
    tmp370 = tmp369 * tmp49
    tmp371 = tmp366 + tmp370
    tmp372 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (0))), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp373 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp374 = tl.where(tmp56, tmp372, tmp373)
    tmp375 = tl.full(tmp374.shape, 0.0, tmp374.dtype)
    tmp376 = tl.where(tmp3, tmp374, tmp375)
    tmp377 = tl.load(in_ptr0 + (122880 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp378 = tl.where(tmp3, tmp376, tmp377)
    tmp379 = tmp378 * tmp12
    tmp380 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (1))), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp381 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp382 = tl.where(tmp76, tmp380, tmp381)
    tmp383 = tl.full(tmp382.shape, 0.0, tmp382.dtype)
    tmp384 = tl.where(tmp16, tmp382, tmp383)
    tmp385 = tl.load(in_ptr0 + (122880 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp386 = tl.where(tmp16, tmp384, tmp385)
    tmp387 = tmp386 * tmp24
    tmp388 = tmp379 + tmp387
    tmp389 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (2))), tmp97 & xmask, eviction_policy='evict_last', other=0.0)
    tmp390 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp391 = tl.where(tmp96, tmp389, tmp390)
    tmp392 = tl.full(tmp391.shape, 0.0, tmp391.dtype)
    tmp393 = tl.where(tmp29, tmp391, tmp392)
    tmp394 = tl.load(in_ptr0 + (122880 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp395 = tl.where(tmp29, tmp393, tmp394)
    tmp396 = tmp395 * tmp37
    tmp397 = tmp388 + tmp396
    tmp398 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (3))), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp399 = tl.load(in_ptr0 + (110592 + x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp400 = tl.where(tmp117, tmp398, tmp399)
    tmp401 = tl.full(tmp400.shape, 0.0, tmp400.dtype)
    tmp402 = tl.where(tmp41, tmp400, tmp401)
    tmp403 = tl.load(in_ptr0 + (122880 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp404 = tl.where(tmp41, tmp402, tmp403)
    tmp405 = tmp404 * tmp49
    tmp406 = tmp397 + tmp405
    tmp407 = tmp371 + tmp133
    tmp408 = -tmp407
    tmp409 = libdevice.exp(tmp408)
    tmp410 = tmp409 + tmp137
    tmp411 = (tmp407 / tmp410)
    tmp412 = tmp406 + tmp133
    tmp413 = -tmp412
    tmp414 = libdevice.exp(tmp413)
    tmp415 = tmp414 + tmp137
    tmp416 = (tmp412 / tmp415)
    tmp417 = tl.load(in_ptr6 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp418 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp419 = tl.where(tmp3, tmp417, tmp418)
    tmp420 = tmp419 * tmp12
    tmp421 = tl.load(in_ptr6 + (1 + 4*x2 + (1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp422 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp423 = tl.where(tmp16, tmp421, tmp422)
    tmp424 = tmp423 * tmp24
    tmp425 = tmp420 + tmp424
    tmp426 = tl.load(in_ptr6 + (1 + 4*x2 + (2)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp427 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp428 = tl.where(tmp29, tmp426, tmp427)
    tmp429 = tmp428 * tmp37
    tmp430 = tmp425 + tmp429
    tmp431 = tl.load(in_ptr6 + (1 + 4*x2 + (3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp432 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp433 = tl.where(tmp41, tmp431, tmp432)
    tmp434 = tmp433 * tmp49
    tmp435 = tmp430 + tmp434
    tmp436 = tl.load(in_ptr6 + (1 + 4*x2 + (1 + (0))), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp437 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp64 & xmask, other=0.0)
    tmp438 = tl.where(tmp56, tmp436, tmp437)
    tmp439 = tl.full(tmp438.shape, 0.0, tmp438.dtype)
    tmp440 = tl.where(tmp3, tmp438, tmp439)
    tmp441 = tl.load(in_ptr0 + (159744 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp442 = tl.where(tmp3, tmp440, tmp441)
    tmp443 = tmp442 * tmp12
    tmp444 = tl.load(in_ptr6 + (1 + 4*x2 + (1 + (1))), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp445 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp84 & xmask, other=0.0)
    tmp446 = tl.where(tmp76, tmp444, tmp445)
    tmp447 = tl.full(tmp446.shape, 0.0, tmp446.dtype)
    tmp448 = tl.where(tmp16, tmp446, tmp447)
    tmp449 = tl.load(in_ptr0 + (159744 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp450 = tl.where(tmp16, tmp448, tmp449)
    tmp451 = tmp450 * tmp24
    tmp452 = tmp443 + tmp451
    tmp453 = tl.load(in_ptr6 + (1 + 4*x2 + (1 + (2))), tmp97 & xmask, eviction_policy='evict_last', other=0.0)
    tmp454 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp104 & xmask, other=0.0)
    tmp455 = tl.where(tmp96, tmp453, tmp454)
    tmp456 = tl.full(tmp455.shape, 0.0, tmp455.dtype)
    tmp457 = tl.where(tmp29, tmp455, tmp456)
    tmp458 = tl.load(in_ptr0 + (159744 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp459 = tl.where(tmp29, tmp457, tmp458)
    tmp460 = tmp459 * tmp37
    tmp461 = tmp452 + tmp460
    tmp462 = tl.load(in_ptr6 + (1 + 4*x2 + (1 + (3))), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp463 = tl.load(in_ptr0 + (147456 + x0 + 3072*x1), tmp124 & xmask, other=0.0)
    tmp464 = tl.where(tmp117, tmp462, tmp463)
    tmp465 = tl.full(tmp464.shape, 0.0, tmp464.dtype)
    tmp466 = tl.where(tmp41, tmp464, tmp465)
    tmp467 = tl.load(in_ptr0 + (159744 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp468 = tl.where(tmp41, tmp466, tmp467)
    tmp469 = tmp468 * tmp49
    tmp470 = tmp461 + tmp469
    tmp471 = tmp435 + tmp133
    tmp472 = -tmp471
    tmp473 = libdevice.exp(tmp472)
    tmp474 = tmp473 + tmp137
    tmp475 = (tmp471 / tmp474)
    tmp476 = tmp470 + tmp133
    tmp477 = -tmp476
    tmp478 = libdevice.exp(tmp477)
    tmp479 = tmp478 + tmp137
    tmp480 = (tmp476 / tmp479)
    tmp481 = tl.load(in_ptr7 + (1 + 4*x2 + (0)), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp482 = tl.load(in_ptr0 + (184320 + x0 + 3072*x1), tmp7 & xmask, other=0.0)
    tmp483 = tl.where(tmp3, tmp481, tmp482)
    tmp484 = tmp483 * tmp12
    tmp485 = tl.load(in_ptr7 + (1 + 4*x2 + (1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp486 = tl.load(in_ptr0 + (184320 + x0 + 3072*x1), tmp20 & xmask, other=0.0)
    tmp487 = tl.where(tmp16, tmp485, tmp486)
    tmp488 = tmp487 * tmp24
    tmp489 = tmp484 + tmp488
    tmp490 = tl.load(in_ptr7 + (1 + 4*x2 + (2)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp491 = tl.load(in_ptr0 + (184320 + x0 + 3072*x1), tmp33 & xmask, other=0.0)
    tmp492 = tl.where(tmp29, tmp490, tmp491)
    tmp493 = tmp492 * tmp37
    tmp494 = tmp489 + tmp493
    tmp495 = tl.load(in_ptr7 + (1 + 4*x2 + (3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp496 = tl.load(in_ptr0 + (184320 + x0 + 3072*x1), tmp45 & xmask, other=0.0)
    tmp497 = tl.where(tmp41, tmp495, tmp496)
    tmp498 = tmp497 * tmp49
    tmp499 = tmp494 + tmp498
    tmp500 = tmp499 + tmp133
    tmp501 = -tmp500
    tmp502 = libdevice.exp(tmp501)
    tmp503 = tmp502 + tmp137
    tmp504 = (tmp500 / tmp503)
    tl.store(in_out_ptr0 + (x2), tmp139, xmask)
    tl.store(in_out_ptr1 + (x2), tmp144, xmask)
    tl.store(in_out_ptr2 + (x2), tmp208, xmask)
    tl.store(in_out_ptr3 + (x2), tmp212, xmask)
    tl.store(in_out_ptr4 + (x2), tmp216, xmask)
    tl.store(in_out_ptr5 + (x2), tmp220, xmask)
    tl.store(in_out_ptr6 + (x2), tmp224, xmask)
    tl.store(in_out_ptr7 + (x2), tmp283, xmask)
    tl.store(in_out_ptr8 + (x2), tmp288, xmask)
    tl.store(in_out_ptr9 + (x2), tmp347, xmask)
    tl.store(in_out_ptr10 + (x2), tmp352, xmask)
    tl.store(in_out_ptr11 + (x2), tmp411, xmask)
    tl.store(in_out_ptr12 + (x2), tmp416, xmask)
    tl.store(in_out_ptr13 + (x2), tmp475, xmask)
    tl.store(in_out_ptr14 + (x2), tmp480, xmask)
    tl.store(in_out_ptr15 + (x2), tmp504, xmask)
