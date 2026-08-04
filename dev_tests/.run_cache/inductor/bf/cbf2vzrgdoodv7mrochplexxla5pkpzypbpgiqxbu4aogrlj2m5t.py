
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_out_ptr9': '*fp32', 'in_out_ptr10': '*fp32', 'in_out_ptr11': '*fp32', 'in_out_ptr12': '*fp32', 'in_out_ptr13': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr10', 'in_out_ptr11', 'in_out_ptr12', 'in_out_ptr13', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 113, 'num_store': 14, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 915456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_silu_slice_split_sum_unsqueeze_28(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_out_ptr10, in_out_ptr11, in_out_ptr12, in_out_ptr13, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
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
    tmp41 = tl.load(in_ptr5 + (4*x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr5 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr6 + (4*x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr6 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (3 + 4*x2), xmask, eviction_policy='evict_last')
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
    tmp42 = tmp41 * tmp1
    tmp44 = tmp43 * tmp4
    tmp45 = tmp42 + tmp44
    tmp47 = tmp46 * tmp8
    tmp48 = tmp45 + tmp47
    tmp50 = tmp49 * tmp12
    tmp51 = tmp48 + tmp50
    tmp52 = tmp51 + tmp15
    tmp54 = tmp53 * tmp1
    tmp56 = tmp55 * tmp4
    tmp57 = tmp54 + tmp56
    tmp59 = tmp58 * tmp8
    tmp60 = tmp57 + tmp59
    tmp62 = tmp61 * tmp12
    tmp63 = tmp60 + tmp62
    tmp64 = tmp63 + tmp15
    tmp65 = -tmp16
    tmp66 = libdevice.exp(tmp65)
    tmp67 = tl.full([1], 1.0, tl.float32)
    tmp68 = tmp66 + tmp67
    tmp69 = (tmp16 / tmp68)
    tmp70 = -tmp28
    tmp71 = libdevice.exp(tmp70)
    tmp72 = tmp71 + tmp67
    tmp73 = (tmp28 / tmp72)
    tmp74 = -tmp40
    tmp75 = libdevice.exp(tmp74)
    tmp76 = tmp75 + tmp67
    tmp77 = (tmp40 / tmp76)
    tmp78 = -tmp52
    tmp79 = libdevice.exp(tmp78)
    tmp80 = tmp79 + tmp67
    tmp81 = (tmp52 / tmp80)
    tmp82 = -tmp64
    tmp83 = libdevice.exp(tmp82)
    tmp84 = tmp83 + tmp67
    tmp85 = (tmp64 / tmp84)
    tmp86 = tl.full([1], 0, tl.int64)
    tmp87 = tmp86 >= tmp86
    tmp88 = tl.full([1], 3, tl.int64)
    tmp89 = tmp86 < tmp88
    tmp90 = tl.load(in_ptr0 + (1 + 4*x2 + (0)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp86 >= tmp88
    tmp92 = tl.full([1], 4, tl.int64)
    tmp93 = tmp86 < tmp92
    tmp94 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp95 = tl.where(tmp89, tmp90, tmp94)
    tmp96 = tmp95 * tmp1
    tmp97 = tl.full([1], 1, tl.int64)
    tmp98 = tmp97 >= tmp86
    tmp99 = tmp97 < tmp88
    tmp100 = tl.load(in_ptr0 + (1 + 4*x2 + (1)), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp97 >= tmp88
    tmp102 = tmp97 < tmp92
    tmp103 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp104 = tl.where(tmp99, tmp100, tmp103)
    tmp105 = tmp104 * tmp4
    tmp106 = tmp96 + tmp105
    tmp107 = tl.full([1], 2, tl.int64)
    tmp108 = tmp107 >= tmp86
    tmp109 = tmp107 < tmp88
    tmp110 = tl.load(in_ptr0 + (1 + 4*x2 + (2)), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp107 >= tmp88
    tmp112 = tmp107 < tmp92
    tmp113 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp114 = tl.where(tmp109, tmp110, tmp113)
    tmp115 = tmp114 * tmp8
    tmp116 = tmp106 + tmp115
    tmp117 = tmp88 >= tmp86
    tmp118 = tmp88 < tmp88
    tmp119 = tl.load(in_ptr0 + (1 + 4*x2 + (3)), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp88 >= tmp88
    tmp121 = tmp88 < tmp92
    tmp122 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp123 = tl.where(tmp118, tmp119, tmp122)
    tmp124 = tmp123 * tmp12
    tmp125 = tmp116 + tmp124
    tmp126 = tl.full([1], 1, tl.int64)
    tmp127 = tl.full([1], 0, tl.int64)
    tmp128 = tmp126 >= tmp127
    tmp129 = tl.full([1], 3, tl.int64)
    tmp130 = tmp126 < tmp129
    tmp131 = tmp130 & tmp89
    tmp132 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (0))), tmp131 & xmask, eviction_policy='evict_last', other=0.0)
    tmp133 = tmp126 >= tmp129
    tmp134 = tl.full([1], 4, tl.int64)
    tmp135 = tmp126 < tmp134
    tmp136 = tmp133 & tmp89
    tmp137 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp136 & xmask, other=0.0)
    tmp138 = tl.where(tmp130, tmp132, tmp137)
    tmp139 = tl.full(tmp138.shape, 0.0, tmp138.dtype)
    tmp140 = tl.where(tmp89, tmp138, tmp139)
    tmp141 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp142 = tl.where(tmp89, tmp140, tmp141)
    tmp143 = tmp142 * tmp1
    tmp144 = tl.full([1], 2, tl.int64)
    tmp145 = tl.full([1], 0, tl.int64)
    tmp146 = tmp144 >= tmp145
    tmp147 = tl.full([1], 3, tl.int64)
    tmp148 = tmp144 < tmp147
    tmp149 = tmp148 & tmp99
    tmp150 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (1))), tmp149 & xmask, eviction_policy='evict_last', other=0.0)
    tmp151 = tmp144 >= tmp147
    tmp152 = tl.full([1], 4, tl.int64)
    tmp153 = tmp144 < tmp152
    tmp154 = tmp151 & tmp99
    tmp155 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp154 & xmask, other=0.0)
    tmp156 = tl.where(tmp148, tmp150, tmp155)
    tmp157 = tl.full(tmp156.shape, 0.0, tmp156.dtype)
    tmp158 = tl.where(tmp99, tmp156, tmp157)
    tmp159 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp160 = tl.where(tmp99, tmp158, tmp159)
    tmp161 = tmp160 * tmp4
    tmp162 = tmp143 + tmp161
    tmp163 = tl.full([1], 3, tl.int64)
    tmp164 = tl.full([1], 0, tl.int64)
    tmp165 = tmp163 >= tmp164
    tmp166 = tmp163 < tmp163
    tmp167 = tmp166 & tmp109
    tmp168 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (2))), tmp167 & xmask, eviction_policy='evict_last', other=0.0)
    tmp169 = tmp163 >= tmp163
    tmp170 = tl.full([1], 4, tl.int64)
    tmp171 = tmp163 < tmp170
    tmp172 = tmp169 & tmp109
    tmp173 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp172 & xmask, other=0.0)
    tmp174 = tl.where(tmp166, tmp168, tmp173)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp109, tmp174, tmp175)
    tmp177 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp178 = tl.where(tmp109, tmp176, tmp177)
    tmp179 = tmp178 * tmp8
    tmp180 = tmp162 + tmp179
    tmp181 = tl.full([1], 4, tl.int64)
    tmp182 = tl.full([1], 0, tl.int64)
    tmp183 = tmp181 >= tmp182
    tmp184 = tl.full([1], 3, tl.int64)
    tmp185 = tmp181 < tmp184
    tmp186 = tmp185 & tmp118
    tmp187 = tl.load(in_ptr0 + (1 + 4*x2 + (1 + (3))), tmp186 & xmask, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp181 >= tmp184
    tmp189 = tmp181 < tmp181
    tmp190 = tmp188 & tmp118
    tmp191 = tl.load(in_ptr7 + (x0 + 3072*x1), tmp190 & xmask, other=0.0)
    tmp192 = tl.where(tmp185, tmp187, tmp191)
    tmp193 = tl.full(tmp192.shape, 0.0, tmp192.dtype)
    tmp194 = tl.where(tmp118, tmp192, tmp193)
    tmp195 = tl.load(in_ptr8 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp196 = tl.where(tmp118, tmp194, tmp195)
    tmp197 = tmp196 * tmp12
    tmp198 = tmp180 + tmp197
    tmp199 = tmp125 + tmp15
    tmp200 = -tmp199
    tmp201 = libdevice.exp(tmp200)
    tmp202 = tmp201 + tmp67
    tmp203 = (tmp199 / tmp202)
    tmp204 = tmp198 + tmp15
    tmp205 = -tmp204
    tmp206 = libdevice.exp(tmp205)
    tmp207 = tmp206 + tmp67
    tmp208 = (tmp204 / tmp207)
    tmp209 = tl.load(in_ptr3 + (1 + 4*x2 + (0)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp210 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp211 = tl.where(tmp89, tmp209, tmp210)
    tmp212 = tmp211 * tmp1
    tmp213 = tl.load(in_ptr3 + (1 + 4*x2 + (1)), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp214 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp215 = tl.where(tmp99, tmp213, tmp214)
    tmp216 = tmp215 * tmp4
    tmp217 = tmp212 + tmp216
    tmp218 = tl.load(in_ptr3 + (1 + 4*x2 + (2)), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
    tmp219 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp220 = tl.where(tmp109, tmp218, tmp219)
    tmp221 = tmp220 * tmp8
    tmp222 = tmp217 + tmp221
    tmp223 = tl.load(in_ptr3 + (1 + 4*x2 + (3)), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp224 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp225 = tl.where(tmp118, tmp223, tmp224)
    tmp226 = tmp225 * tmp12
    tmp227 = tmp222 + tmp226
    tmp228 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (0))), tmp131 & xmask, eviction_policy='evict_last', other=0.0)
    tmp229 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp136 & xmask, other=0.0)
    tmp230 = tl.where(tmp130, tmp228, tmp229)
    tmp231 = tl.full(tmp230.shape, 0.0, tmp230.dtype)
    tmp232 = tl.where(tmp89, tmp230, tmp231)
    tmp233 = tl.load(in_ptr10 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp234 = tl.where(tmp89, tmp232, tmp233)
    tmp235 = tmp234 * tmp1
    tmp236 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (1))), tmp149 & xmask, eviction_policy='evict_last', other=0.0)
    tmp237 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp154 & xmask, other=0.0)
    tmp238 = tl.where(tmp148, tmp236, tmp237)
    tmp239 = tl.full(tmp238.shape, 0.0, tmp238.dtype)
    tmp240 = tl.where(tmp99, tmp238, tmp239)
    tmp241 = tl.load(in_ptr10 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp242 = tl.where(tmp99, tmp240, tmp241)
    tmp243 = tmp242 * tmp4
    tmp244 = tmp235 + tmp243
    tmp245 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (2))), tmp167 & xmask, eviction_policy='evict_last', other=0.0)
    tmp246 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp172 & xmask, other=0.0)
    tmp247 = tl.where(tmp166, tmp245, tmp246)
    tmp248 = tl.full(tmp247.shape, 0.0, tmp247.dtype)
    tmp249 = tl.where(tmp109, tmp247, tmp248)
    tmp250 = tl.load(in_ptr10 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp251 = tl.where(tmp109, tmp249, tmp250)
    tmp252 = tmp251 * tmp8
    tmp253 = tmp244 + tmp252
    tmp254 = tl.load(in_ptr3 + (1 + 4*x2 + (1 + (3))), tmp186 & xmask, eviction_policy='evict_last', other=0.0)
    tmp255 = tl.load(in_ptr9 + (x0 + 3072*x1), tmp190 & xmask, other=0.0)
    tmp256 = tl.where(tmp185, tmp254, tmp255)
    tmp257 = tl.full(tmp256.shape, 0.0, tmp256.dtype)
    tmp258 = tl.where(tmp118, tmp256, tmp257)
    tmp259 = tl.load(in_ptr10 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp260 = tl.where(tmp118, tmp258, tmp259)
    tmp261 = tmp260 * tmp12
    tmp262 = tmp253 + tmp261
    tmp263 = tmp227 + tmp15
    tmp264 = -tmp263
    tmp265 = libdevice.exp(tmp264)
    tmp266 = tmp265 + tmp67
    tmp267 = (tmp263 / tmp266)
    tmp268 = tmp262 + tmp15
    tmp269 = -tmp268
    tmp270 = libdevice.exp(tmp269)
    tmp271 = tmp270 + tmp67
    tmp272 = (tmp268 / tmp271)
    tmp273 = tl.load(in_ptr4 + (1 + 4*x2 + (0)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp274 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp275 = tl.where(tmp89, tmp273, tmp274)
    tmp276 = tmp275 * tmp1
    tmp277 = tl.load(in_ptr4 + (1 + 4*x2 + (1)), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp278 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp279 = tl.where(tmp99, tmp277, tmp278)
    tmp280 = tmp279 * tmp4
    tmp281 = tmp276 + tmp280
    tmp282 = tl.load(in_ptr4 + (1 + 4*x2 + (2)), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
    tmp283 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp284 = tl.where(tmp109, tmp282, tmp283)
    tmp285 = tmp284 * tmp8
    tmp286 = tmp281 + tmp285
    tmp287 = tl.load(in_ptr4 + (1 + 4*x2 + (3)), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp288 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp289 = tl.where(tmp118, tmp287, tmp288)
    tmp290 = tmp289 * tmp12
    tmp291 = tmp286 + tmp290
    tmp292 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (0))), tmp131 & xmask, eviction_policy='evict_last', other=0.0)
    tmp293 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp136 & xmask, other=0.0)
    tmp294 = tl.where(tmp130, tmp292, tmp293)
    tmp295 = tl.full(tmp294.shape, 0.0, tmp294.dtype)
    tmp296 = tl.where(tmp89, tmp294, tmp295)
    tmp297 = tl.load(in_ptr12 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp298 = tl.where(tmp89, tmp296, tmp297)
    tmp299 = tmp298 * tmp1
    tmp300 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (1))), tmp149 & xmask, eviction_policy='evict_last', other=0.0)
    tmp301 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp154 & xmask, other=0.0)
    tmp302 = tl.where(tmp148, tmp300, tmp301)
    tmp303 = tl.full(tmp302.shape, 0.0, tmp302.dtype)
    tmp304 = tl.where(tmp99, tmp302, tmp303)
    tmp305 = tl.load(in_ptr12 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp306 = tl.where(tmp99, tmp304, tmp305)
    tmp307 = tmp306 * tmp4
    tmp308 = tmp299 + tmp307
    tmp309 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (2))), tmp167 & xmask, eviction_policy='evict_last', other=0.0)
    tmp310 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp172 & xmask, other=0.0)
    tmp311 = tl.where(tmp166, tmp309, tmp310)
    tmp312 = tl.full(tmp311.shape, 0.0, tmp311.dtype)
    tmp313 = tl.where(tmp109, tmp311, tmp312)
    tmp314 = tl.load(in_ptr12 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp315 = tl.where(tmp109, tmp313, tmp314)
    tmp316 = tmp315 * tmp8
    tmp317 = tmp308 + tmp316
    tmp318 = tl.load(in_ptr4 + (1 + 4*x2 + (1 + (3))), tmp186 & xmask, eviction_policy='evict_last', other=0.0)
    tmp319 = tl.load(in_ptr11 + (x0 + 3072*x1), tmp190 & xmask, other=0.0)
    tmp320 = tl.where(tmp185, tmp318, tmp319)
    tmp321 = tl.full(tmp320.shape, 0.0, tmp320.dtype)
    tmp322 = tl.where(tmp118, tmp320, tmp321)
    tmp323 = tl.load(in_ptr12 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp324 = tl.where(tmp118, tmp322, tmp323)
    tmp325 = tmp324 * tmp12
    tmp326 = tmp317 + tmp325
    tmp327 = tmp291 + tmp15
    tmp328 = -tmp327
    tmp329 = libdevice.exp(tmp328)
    tmp330 = tmp329 + tmp67
    tmp331 = (tmp327 / tmp330)
    tmp332 = tmp326 + tmp15
    tmp333 = -tmp332
    tmp334 = libdevice.exp(tmp333)
    tmp335 = tmp334 + tmp67
    tmp336 = (tmp332 / tmp335)
    tmp337 = tl.load(in_ptr5 + (1 + 4*x2 + (0)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp338 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp339 = tl.where(tmp89, tmp337, tmp338)
    tmp340 = tmp339 * tmp1
    tmp341 = tl.load(in_ptr5 + (1 + 4*x2 + (1)), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp342 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp343 = tl.where(tmp99, tmp341, tmp342)
    tmp344 = tmp343 * tmp4
    tmp345 = tmp340 + tmp344
    tmp346 = tl.load(in_ptr5 + (1 + 4*x2 + (2)), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
    tmp347 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp348 = tl.where(tmp109, tmp346, tmp347)
    tmp349 = tmp348 * tmp8
    tmp350 = tmp345 + tmp349
    tmp351 = tl.load(in_ptr5 + (1 + 4*x2 + (3)), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp352 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp353 = tl.where(tmp118, tmp351, tmp352)
    tmp354 = tmp353 * tmp12
    tmp355 = tmp350 + tmp354
    tmp356 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (0))), tmp131 & xmask, eviction_policy='evict_last', other=0.0)
    tmp357 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp136 & xmask, other=0.0)
    tmp358 = tl.where(tmp130, tmp356, tmp357)
    tmp359 = tl.full(tmp358.shape, 0.0, tmp358.dtype)
    tmp360 = tl.where(tmp89, tmp358, tmp359)
    tmp361 = tl.load(in_ptr14 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp362 = tl.where(tmp89, tmp360, tmp361)
    tmp363 = tmp362 * tmp1
    tmp364 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (1))), tmp149 & xmask, eviction_policy='evict_last', other=0.0)
    tmp365 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp154 & xmask, other=0.0)
    tmp366 = tl.where(tmp148, tmp364, tmp365)
    tmp367 = tl.full(tmp366.shape, 0.0, tmp366.dtype)
    tmp368 = tl.where(tmp99, tmp366, tmp367)
    tmp369 = tl.load(in_ptr14 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp370 = tl.where(tmp99, tmp368, tmp369)
    tmp371 = tmp370 * tmp4
    tmp372 = tmp363 + tmp371
    tmp373 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (2))), tmp167 & xmask, eviction_policy='evict_last', other=0.0)
    tmp374 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp172 & xmask, other=0.0)
    tmp375 = tl.where(tmp166, tmp373, tmp374)
    tmp376 = tl.full(tmp375.shape, 0.0, tmp375.dtype)
    tmp377 = tl.where(tmp109, tmp375, tmp376)
    tmp378 = tl.load(in_ptr14 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp379 = tl.where(tmp109, tmp377, tmp378)
    tmp380 = tmp379 * tmp8
    tmp381 = tmp372 + tmp380
    tmp382 = tl.load(in_ptr5 + (1 + 4*x2 + (1 + (3))), tmp186 & xmask, eviction_policy='evict_last', other=0.0)
    tmp383 = tl.load(in_ptr13 + (x0 + 3072*x1), tmp190 & xmask, other=0.0)
    tmp384 = tl.where(tmp185, tmp382, tmp383)
    tmp385 = tl.full(tmp384.shape, 0.0, tmp384.dtype)
    tmp386 = tl.where(tmp118, tmp384, tmp385)
    tmp387 = tl.load(in_ptr14 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp388 = tl.where(tmp118, tmp386, tmp387)
    tmp389 = tmp388 * tmp12
    tmp390 = tmp381 + tmp389
    tmp391 = tmp355 + tmp15
    tmp392 = -tmp391
    tmp393 = libdevice.exp(tmp392)
    tmp394 = tmp393 + tmp67
    tmp395 = (tmp391 / tmp394)
    tmp396 = tmp390 + tmp15
    tmp397 = -tmp396
    tmp398 = libdevice.exp(tmp397)
    tmp399 = tmp398 + tmp67
    tmp400 = (tmp396 / tmp399)
    tmp401 = tl.load(in_ptr6 + (1 + 4*x2 + (0)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp402 = tl.load(in_ptr15 + (x0 + 3072*x1), tmp91 & xmask, other=0.0)
    tmp403 = tl.where(tmp89, tmp401, tmp402)
    tmp404 = tmp403 * tmp1
    tmp405 = tl.load(in_ptr6 + (1 + 4*x2 + (1)), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp406 = tl.load(in_ptr15 + (x0 + 3072*x1), tmp101 & xmask, other=0.0)
    tmp407 = tl.where(tmp99, tmp405, tmp406)
    tmp408 = tmp407 * tmp4
    tmp409 = tmp404 + tmp408
    tmp410 = tl.load(in_ptr6 + (1 + 4*x2 + (2)), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
    tmp411 = tl.load(in_ptr15 + (x0 + 3072*x1), tmp111 & xmask, other=0.0)
    tmp412 = tl.where(tmp109, tmp410, tmp411)
    tmp413 = tmp412 * tmp8
    tmp414 = tmp409 + tmp413
    tmp415 = tl.load(in_ptr6 + (1 + 4*x2 + (3)), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp416 = tl.load(in_ptr15 + (x0 + 3072*x1), tmp120 & xmask, other=0.0)
    tmp417 = tl.where(tmp118, tmp415, tmp416)
    tmp418 = tmp417 * tmp12
    tmp419 = tmp414 + tmp418
    tmp420 = tmp419 + tmp15
    tmp421 = -tmp420
    tmp422 = libdevice.exp(tmp421)
    tmp423 = tmp422 + tmp67
    tmp424 = (tmp420 / tmp423)
    tl.store(in_out_ptr0 + (x2), tmp69, xmask)
    tl.store(in_out_ptr1 + (x2), tmp73, xmask)
    tl.store(in_out_ptr2 + (x2), tmp77, xmask)
    tl.store(in_out_ptr3 + (x2), tmp81, xmask)
    tl.store(in_out_ptr4 + (x2), tmp85, xmask)
    tl.store(in_out_ptr5 + (x2), tmp203, xmask)
    tl.store(in_out_ptr6 + (x2), tmp208, xmask)
    tl.store(in_out_ptr7 + (x2), tmp267, xmask)
    tl.store(in_out_ptr8 + (x2), tmp272, xmask)
    tl.store(in_out_ptr9 + (x2), tmp331, xmask)
    tl.store(in_out_ptr10 + (x2), tmp336, xmask)
    tl.store(in_out_ptr11 + (x2), tmp395, xmask)
    tl.store(in_out_ptr12 + (x2), tmp400, xmask)
    tl.store(in_out_ptr13 + (x2), tmp424, xmask)
