
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_out_ptr9': '*fp32', 'in_out_ptr10': '*fp32', 'in_out_ptr11': '*fp32', 'in_out_ptr12': '*fp32', 'in_out_ptr13': '*fp32', 'in_out_ptr14': '*fp32', 'in_out_ptr15': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr10', 'in_out_ptr11', 'in_out_ptr12', 'in_out_ptr13', 'in_out_ptr14', 'in_out_ptr15', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 18, 'num_store': 16, 'num_reduction': 64, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 595968}}
)
@triton.jit
def triton_per_fused_native_layer_norm_24(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_out_ptr10, in_out_ptr11, in_out_ptr12, in_out_ptr13, in_out_ptr14, in_out_ptr15, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp32 = tl.load(in_out_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp47 = tl.load(in_out_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp62 = tl.load(in_out_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp77 = tl.load(in_out_ptr5 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp92 = tl.load(in_out_ptr6 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp107 = tl.load(in_out_ptr7 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp122 = tl.load(in_out_ptr8 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp137 = tl.load(in_out_ptr9 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp152 = tl.load(in_out_ptr10 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp167 = tl.load(in_out_ptr11 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp182 = tl.load(in_out_ptr12 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp197 = tl.load(in_out_ptr13 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp212 = tl.load(in_out_ptr14 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp227 = tl.load(in_out_ptr15 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp249 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp251 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = (tmp24 / tmp9)
    tmp26 = tmp18 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
    tmp35 = tl.where(r0_mask & xmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp38 = tl.where(r0_mask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None].to(tl.float32)
    tmp40 = (tmp39 / tmp9)
    tmp41 = tmp33 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
    tmp45 = tl.where(r0_mask & xmask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask & xmask, tmp48, 0)
    tmp51 = tl.broadcast_to(tmp48, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(r0_mask & xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None].to(tl.float32)
    tmp55 = (tmp54 / tmp9)
    tmp56 = tmp48 - tmp55
    tmp57 = tmp56 * tmp56
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
    tmp60 = tl.where(r0_mask & xmask, tmp58, 0)
    tmp61 = tl.sum(tmp60, 1)[:, None].to(tl.float32)
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK, R0_BLOCK])
    tmp65 = tl.where(r0_mask & xmask, tmp63, 0)
    tmp66 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(r0_mask & xmask, tmp66, 0)
    tmp69 = tl.sum(tmp68, 1)[:, None].to(tl.float32)
    tmp70 = (tmp69 / tmp9)
    tmp71 = tmp63 - tmp70
    tmp72 = tmp71 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, R0_BLOCK])
    tmp75 = tl.where(r0_mask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None].to(tl.float32)
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, R0_BLOCK])
    tmp80 = tl.where(r0_mask & xmask, tmp78, 0)
    tmp81 = tl.broadcast_to(tmp78, [XBLOCK, R0_BLOCK])
    tmp83 = tl.where(r0_mask & xmask, tmp81, 0)
    tmp84 = tl.sum(tmp83, 1)[:, None].to(tl.float32)
    tmp85 = (tmp84 / tmp9)
    tmp86 = tmp78 - tmp85
    tmp87 = tmp86 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, R0_BLOCK])
    tmp90 = tl.where(r0_mask & xmask, tmp88, 0)
    tmp91 = tl.sum(tmp90, 1)[:, None].to(tl.float32)
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, R0_BLOCK])
    tmp95 = tl.where(r0_mask & xmask, tmp93, 0)
    tmp96 = tl.broadcast_to(tmp93, [XBLOCK, R0_BLOCK])
    tmp98 = tl.where(r0_mask & xmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None].to(tl.float32)
    tmp100 = (tmp99 / tmp9)
    tmp101 = tmp93 - tmp100
    tmp102 = tmp101 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, R0_BLOCK])
    tmp105 = tl.where(r0_mask & xmask, tmp103, 0)
    tmp106 = tl.sum(tmp105, 1)[:, None].to(tl.float32)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, R0_BLOCK])
    tmp110 = tl.where(r0_mask & xmask, tmp108, 0)
    tmp111 = tl.broadcast_to(tmp108, [XBLOCK, R0_BLOCK])
    tmp113 = tl.where(r0_mask & xmask, tmp111, 0)
    tmp114 = tl.sum(tmp113, 1)[:, None].to(tl.float32)
    tmp115 = (tmp114 / tmp9)
    tmp116 = tmp108 - tmp115
    tmp117 = tmp116 * tmp116
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, R0_BLOCK])
    tmp120 = tl.where(r0_mask & xmask, tmp118, 0)
    tmp121 = tl.sum(tmp120, 1)[:, None].to(tl.float32)
    tmp123 = tl.broadcast_to(tmp122, [XBLOCK, R0_BLOCK])
    tmp125 = tl.where(r0_mask & xmask, tmp123, 0)
    tmp126 = tl.broadcast_to(tmp123, [XBLOCK, R0_BLOCK])
    tmp128 = tl.where(r0_mask & xmask, tmp126, 0)
    tmp129 = tl.sum(tmp128, 1)[:, None].to(tl.float32)
    tmp130 = (tmp129 / tmp9)
    tmp131 = tmp123 - tmp130
    tmp132 = tmp131 * tmp131
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, R0_BLOCK])
    tmp135 = tl.where(r0_mask & xmask, tmp133, 0)
    tmp136 = tl.sum(tmp135, 1)[:, None].to(tl.float32)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, R0_BLOCK])
    tmp140 = tl.where(r0_mask & xmask, tmp138, 0)
    tmp141 = tl.broadcast_to(tmp138, [XBLOCK, R0_BLOCK])
    tmp143 = tl.where(r0_mask & xmask, tmp141, 0)
    tmp144 = tl.sum(tmp143, 1)[:, None].to(tl.float32)
    tmp145 = (tmp144 / tmp9)
    tmp146 = tmp138 - tmp145
    tmp147 = tmp146 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, R0_BLOCK])
    tmp150 = tl.where(r0_mask & xmask, tmp148, 0)
    tmp151 = tl.sum(tmp150, 1)[:, None].to(tl.float32)
    tmp153 = tl.broadcast_to(tmp152, [XBLOCK, R0_BLOCK])
    tmp155 = tl.where(r0_mask & xmask, tmp153, 0)
    tmp156 = tl.broadcast_to(tmp153, [XBLOCK, R0_BLOCK])
    tmp158 = tl.where(r0_mask & xmask, tmp156, 0)
    tmp159 = tl.sum(tmp158, 1)[:, None].to(tl.float32)
    tmp160 = (tmp159 / tmp9)
    tmp161 = tmp153 - tmp160
    tmp162 = tmp161 * tmp161
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, R0_BLOCK])
    tmp165 = tl.where(r0_mask & xmask, tmp163, 0)
    tmp166 = tl.sum(tmp165, 1)[:, None].to(tl.float32)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, R0_BLOCK])
    tmp170 = tl.where(r0_mask & xmask, tmp168, 0)
    tmp171 = tl.broadcast_to(tmp168, [XBLOCK, R0_BLOCK])
    tmp173 = tl.where(r0_mask & xmask, tmp171, 0)
    tmp174 = tl.sum(tmp173, 1)[:, None].to(tl.float32)
    tmp175 = (tmp174 / tmp9)
    tmp176 = tmp168 - tmp175
    tmp177 = tmp176 * tmp176
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, R0_BLOCK])
    tmp180 = tl.where(r0_mask & xmask, tmp178, 0)
    tmp181 = tl.sum(tmp180, 1)[:, None].to(tl.float32)
    tmp183 = tl.broadcast_to(tmp182, [XBLOCK, R0_BLOCK])
    tmp185 = tl.where(r0_mask & xmask, tmp183, 0)
    tmp186 = tl.broadcast_to(tmp183, [XBLOCK, R0_BLOCK])
    tmp188 = tl.where(r0_mask & xmask, tmp186, 0)
    tmp189 = tl.sum(tmp188, 1)[:, None].to(tl.float32)
    tmp190 = (tmp189 / tmp9)
    tmp191 = tmp183 - tmp190
    tmp192 = tmp191 * tmp191
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, R0_BLOCK])
    tmp195 = tl.where(r0_mask & xmask, tmp193, 0)
    tmp196 = tl.sum(tmp195, 1)[:, None].to(tl.float32)
    tmp198 = tl.broadcast_to(tmp197, [XBLOCK, R0_BLOCK])
    tmp200 = tl.where(r0_mask & xmask, tmp198, 0)
    tmp201 = tl.broadcast_to(tmp198, [XBLOCK, R0_BLOCK])
    tmp203 = tl.where(r0_mask & xmask, tmp201, 0)
    tmp204 = tl.sum(tmp203, 1)[:, None].to(tl.float32)
    tmp205 = (tmp204 / tmp9)
    tmp206 = tmp198 - tmp205
    tmp207 = tmp206 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, R0_BLOCK])
    tmp210 = tl.where(r0_mask & xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None].to(tl.float32)
    tmp213 = tl.broadcast_to(tmp212, [XBLOCK, R0_BLOCK])
    tmp215 = tl.where(r0_mask & xmask, tmp213, 0)
    tmp216 = tl.broadcast_to(tmp213, [XBLOCK, R0_BLOCK])
    tmp218 = tl.where(r0_mask & xmask, tmp216, 0)
    tmp219 = tl.sum(tmp218, 1)[:, None].to(tl.float32)
    tmp220 = (tmp219 / tmp9)
    tmp221 = tmp213 - tmp220
    tmp222 = tmp221 * tmp221
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, R0_BLOCK])
    tmp225 = tl.where(r0_mask & xmask, tmp223, 0)
    tmp226 = tl.sum(tmp225, 1)[:, None].to(tl.float32)
    tmp228 = tl.broadcast_to(tmp227, [XBLOCK, R0_BLOCK])
    tmp230 = tl.where(r0_mask & xmask, tmp228, 0)
    tmp231 = tl.broadcast_to(tmp228, [XBLOCK, R0_BLOCK])
    tmp233 = tl.where(r0_mask & xmask, tmp231, 0)
    tmp234 = tl.sum(tmp233, 1)[:, None].to(tl.float32)
    tmp235 = (tmp234 / tmp9)
    tmp236 = tmp228 - tmp235
    tmp237 = tmp236 * tmp236
    tmp238 = tl.broadcast_to(tmp237, [XBLOCK, R0_BLOCK])
    tmp240 = tl.where(r0_mask & xmask, tmp238, 0)
    tmp241 = tl.sum(tmp240, 1)[:, None].to(tl.float32)
    tmp242 = tmp0 - tmp10
    tmp243 = tl.full([1, 1], 768.0, tl.float32)
    tmp244 = (tmp16 / tmp243)
    tmp245 = tl.full([1, 1], 1e-05, tl.float32)
    tmp246 = tmp244 + tmp245
    tmp247 = libdevice.rsqrt(tmp246)
    tmp248 = tmp242 * tmp247
    tmp250 = tmp248 * tmp249
    tmp252 = tmp250 + tmp251
    tmp253 = tmp17 - tmp25
    tmp254 = (tmp31 / tmp243)
    tmp255 = tmp254 + tmp245
    tmp256 = libdevice.rsqrt(tmp255)
    tmp257 = tmp253 * tmp256
    tmp258 = tmp257 * tmp249
    tmp259 = tmp258 + tmp251
    tmp260 = tmp32 - tmp40
    tmp261 = (tmp46 / tmp243)
    tmp262 = tmp261 + tmp245
    tmp263 = libdevice.rsqrt(tmp262)
    tmp264 = tmp260 * tmp263
    tmp265 = tmp264 * tmp249
    tmp266 = tmp265 + tmp251
    tmp267 = tmp47 - tmp55
    tmp268 = (tmp61 / tmp243)
    tmp269 = tmp268 + tmp245
    tmp270 = libdevice.rsqrt(tmp269)
    tmp271 = tmp267 * tmp270
    tmp272 = tmp271 * tmp249
    tmp273 = tmp272 + tmp251
    tmp274 = tmp62 - tmp70
    tmp275 = (tmp76 / tmp243)
    tmp276 = tmp275 + tmp245
    tmp277 = libdevice.rsqrt(tmp276)
    tmp278 = tmp274 * tmp277
    tmp279 = tmp278 * tmp249
    tmp280 = tmp279 + tmp251
    tmp281 = tmp77 - tmp85
    tmp282 = (tmp91 / tmp243)
    tmp283 = tmp282 + tmp245
    tmp284 = libdevice.rsqrt(tmp283)
    tmp285 = tmp281 * tmp284
    tmp286 = tmp285 * tmp249
    tmp287 = tmp286 + tmp251
    tmp288 = tmp92 - tmp100
    tmp289 = (tmp106 / tmp243)
    tmp290 = tmp289 + tmp245
    tmp291 = libdevice.rsqrt(tmp290)
    tmp292 = tmp288 * tmp291
    tmp293 = tmp292 * tmp249
    tmp294 = tmp293 + tmp251
    tmp295 = tmp107 - tmp115
    tmp296 = (tmp121 / tmp243)
    tmp297 = tmp296 + tmp245
    tmp298 = libdevice.rsqrt(tmp297)
    tmp299 = tmp295 * tmp298
    tmp300 = tmp299 * tmp249
    tmp301 = tmp300 + tmp251
    tmp302 = tmp122 - tmp130
    tmp303 = (tmp136 / tmp243)
    tmp304 = tmp303 + tmp245
    tmp305 = libdevice.rsqrt(tmp304)
    tmp306 = tmp302 * tmp305
    tmp307 = tmp306 * tmp249
    tmp308 = tmp307 + tmp251
    tmp309 = tmp137 - tmp145
    tmp310 = (tmp151 / tmp243)
    tmp311 = tmp310 + tmp245
    tmp312 = libdevice.rsqrt(tmp311)
    tmp313 = tmp309 * tmp312
    tmp314 = tmp313 * tmp249
    tmp315 = tmp314 + tmp251
    tmp316 = tmp152 - tmp160
    tmp317 = (tmp166 / tmp243)
    tmp318 = tmp317 + tmp245
    tmp319 = libdevice.rsqrt(tmp318)
    tmp320 = tmp316 * tmp319
    tmp321 = tmp320 * tmp249
    tmp322 = tmp321 + tmp251
    tmp323 = tmp167 - tmp175
    tmp324 = (tmp181 / tmp243)
    tmp325 = tmp324 + tmp245
    tmp326 = libdevice.rsqrt(tmp325)
    tmp327 = tmp323 * tmp326
    tmp328 = tmp327 * tmp249
    tmp329 = tmp328 + tmp251
    tmp330 = tmp182 - tmp190
    tmp331 = (tmp196 / tmp243)
    tmp332 = tmp331 + tmp245
    tmp333 = libdevice.rsqrt(tmp332)
    tmp334 = tmp330 * tmp333
    tmp335 = tmp334 * tmp249
    tmp336 = tmp335 + tmp251
    tmp337 = tmp197 - tmp205
    tmp338 = (tmp211 / tmp243)
    tmp339 = tmp338 + tmp245
    tmp340 = libdevice.rsqrt(tmp339)
    tmp341 = tmp337 * tmp340
    tmp342 = tmp341 * tmp249
    tmp343 = tmp342 + tmp251
    tmp344 = tmp212 - tmp220
    tmp345 = (tmp226 / tmp243)
    tmp346 = tmp345 + tmp245
    tmp347 = libdevice.rsqrt(tmp346)
    tmp348 = tmp344 * tmp347
    tmp349 = tmp348 * tmp249
    tmp350 = tmp349 + tmp251
    tmp351 = tmp227 - tmp235
    tmp352 = (tmp241 / tmp243)
    tmp353 = tmp352 + tmp245
    tmp354 = libdevice.rsqrt(tmp353)
    tmp355 = tmp351 * tmp354
    tmp356 = tmp355 * tmp249
    tmp357 = tmp356 + tmp251
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp252, r0_mask & xmask)
    tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp259, r0_mask & xmask)
    tl.store(in_out_ptr2 + (r0_1 + 768*x0), tmp266, r0_mask & xmask)
    tl.store(in_out_ptr3 + (r0_1 + 768*x0), tmp273, r0_mask & xmask)
    tl.store(in_out_ptr4 + (r0_1 + 768*x0), tmp280, r0_mask & xmask)
    tl.store(in_out_ptr5 + (r0_1 + 768*x0), tmp287, r0_mask & xmask)
    tl.store(in_out_ptr6 + (r0_1 + 768*x0), tmp294, r0_mask & xmask)
    tl.store(in_out_ptr7 + (r0_1 + 768*x0), tmp301, r0_mask & xmask)
    tl.store(in_out_ptr8 + (r0_1 + 768*x0), tmp308, r0_mask & xmask)
    tl.store(in_out_ptr9 + (r0_1 + 768*x0), tmp315, r0_mask & xmask)
    tl.store(in_out_ptr10 + (r0_1 + 768*x0), tmp322, r0_mask & xmask)
    tl.store(in_out_ptr11 + (r0_1 + 768*x0), tmp329, r0_mask & xmask)
    tl.store(in_out_ptr12 + (r0_1 + 768*x0), tmp336, r0_mask & xmask)
    tl.store(in_out_ptr13 + (r0_1 + 768*x0), tmp343, r0_mask & xmask)
    tl.store(in_out_ptr14 + (r0_1 + 768*x0), tmp350, r0_mask & xmask)
    tl.store(in_out_ptr15 + (r0_1 + 768*x0), tmp357, r0_mask & xmask)
