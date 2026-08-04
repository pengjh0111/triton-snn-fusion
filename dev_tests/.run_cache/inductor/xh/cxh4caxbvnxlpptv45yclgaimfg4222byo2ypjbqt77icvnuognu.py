
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_split_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 32, 'num_store': 12, 'num_reduction': 24, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 669696}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_split_50(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (33792 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr0 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr2 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (30720 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp18 = tl.load(in_out_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr2 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr3 + (27648 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp27 = tl.load(in_out_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr1 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr2 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp34 = tl.load(in_ptr3 + (24576 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp36 = tl.load(in_out_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr0 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp39 = tl.load(in_ptr1 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr2 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp43 = tl.load(in_ptr3 + (21504 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp45 = tl.load(in_out_ptr5 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp46 = tl.load(in_ptr0 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr1 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp50 = tl.load(in_ptr2 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp52 = tl.load(in_ptr3 + (18432 + r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp147 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp149 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp44 = tmp42 + tmp43
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp56 = tl.where(r0_mask & xmask, tmp54, 0)
    tmp57 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp59 = tl.where(r0_mask & xmask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None].to(tl.float32)
    tmp61 = tl.full([1, 1], 768, tl.int32)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = (tmp60 / tmp62)
    tmp64 = tmp54 - tmp63
    tmp65 = tmp64 * tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(r0_mask & xmask, tmp66, 0)
    tmp69 = tl.sum(tmp68, 1)[:, None].to(tl.float32)
    tmp70 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp72 = tl.where(r0_mask & xmask, tmp70, 0)
    tmp73 = tl.broadcast_to(tmp70, [XBLOCK, R0_BLOCK])
    tmp75 = tl.where(r0_mask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None].to(tl.float32)
    tmp77 = (tmp76 / tmp62)
    tmp78 = tmp70 - tmp77
    tmp79 = tmp78 * tmp78
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK, R0_BLOCK])
    tmp82 = tl.where(r0_mask & xmask, tmp80, 0)
    tmp83 = tl.sum(tmp82, 1)[:, None].to(tl.float32)
    tmp84 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp86 = tl.where(r0_mask & xmask, tmp84, 0)
    tmp87 = tl.broadcast_to(tmp84, [XBLOCK, R0_BLOCK])
    tmp89 = tl.where(r0_mask & xmask, tmp87, 0)
    tmp90 = tl.sum(tmp89, 1)[:, None].to(tl.float32)
    tmp91 = (tmp90 / tmp62)
    tmp92 = tmp84 - tmp91
    tmp93 = tmp92 * tmp92
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, R0_BLOCK])
    tmp96 = tl.where(r0_mask & xmask, tmp94, 0)
    tmp97 = tl.sum(tmp96, 1)[:, None].to(tl.float32)
    tmp98 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
    tmp100 = tl.where(r0_mask & xmask, tmp98, 0)
    tmp101 = tl.broadcast_to(tmp98, [XBLOCK, R0_BLOCK])
    tmp103 = tl.where(r0_mask & xmask, tmp101, 0)
    tmp104 = tl.sum(tmp103, 1)[:, None].to(tl.float32)
    tmp105 = (tmp104 / tmp62)
    tmp106 = tmp98 - tmp105
    tmp107 = tmp106 * tmp106
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, R0_BLOCK])
    tmp110 = tl.where(r0_mask & xmask, tmp108, 0)
    tmp111 = tl.sum(tmp110, 1)[:, None].to(tl.float32)
    tmp112 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp114 = tl.where(r0_mask & xmask, tmp112, 0)
    tmp115 = tl.broadcast_to(tmp112, [XBLOCK, R0_BLOCK])
    tmp117 = tl.where(r0_mask & xmask, tmp115, 0)
    tmp118 = tl.sum(tmp117, 1)[:, None].to(tl.float32)
    tmp119 = (tmp118 / tmp62)
    tmp120 = tmp112 - tmp119
    tmp121 = tmp120 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK, R0_BLOCK])
    tmp124 = tl.where(r0_mask & xmask, tmp122, 0)
    tmp125 = tl.sum(tmp124, 1)[:, None].to(tl.float32)
    tmp126 = tl.broadcast_to(tmp53, [XBLOCK, R0_BLOCK])
    tmp128 = tl.where(r0_mask & xmask, tmp126, 0)
    tmp129 = tl.broadcast_to(tmp126, [XBLOCK, R0_BLOCK])
    tmp131 = tl.where(r0_mask & xmask, tmp129, 0)
    tmp132 = tl.sum(tmp131, 1)[:, None].to(tl.float32)
    tmp133 = (tmp132 / tmp62)
    tmp134 = tmp126 - tmp133
    tmp135 = tmp134 * tmp134
    tmp136 = tl.broadcast_to(tmp135, [XBLOCK, R0_BLOCK])
    tmp138 = tl.where(r0_mask & xmask, tmp136, 0)
    tmp139 = tl.sum(tmp138, 1)[:, None].to(tl.float32)
    tmp140 = tmp53 - tmp133
    tmp141 = tl.full([1, 1], 768.0, tl.float32)
    tmp142 = (tmp139 / tmp141)
    tmp143 = tl.full([1, 1], 1e-05, tl.float32)
    tmp144 = tmp142 + tmp143
    tmp145 = libdevice.rsqrt(tmp144)
    tmp146 = tmp140 * tmp145
    tmp148 = tmp146 * tmp147
    tmp150 = tmp148 + tmp149
    tmp151 = tmp44 - tmp119
    tmp152 = (tmp125 / tmp141)
    tmp153 = tmp152 + tmp143
    tmp154 = libdevice.rsqrt(tmp153)
    tmp155 = tmp151 * tmp154
    tmp156 = tmp155 * tmp147
    tmp157 = tmp156 + tmp149
    tmp158 = tmp35 - tmp105
    tmp159 = (tmp111 / tmp141)
    tmp160 = tmp159 + tmp143
    tmp161 = libdevice.rsqrt(tmp160)
    tmp162 = tmp158 * tmp161
    tmp163 = tmp162 * tmp147
    tmp164 = tmp163 + tmp149
    tmp165 = tmp26 - tmp91
    tmp166 = (tmp97 / tmp141)
    tmp167 = tmp166 + tmp143
    tmp168 = libdevice.rsqrt(tmp167)
    tmp169 = tmp165 * tmp168
    tmp170 = tmp169 * tmp147
    tmp171 = tmp170 + tmp149
    tmp172 = tmp17 - tmp77
    tmp173 = (tmp83 / tmp141)
    tmp174 = tmp173 + tmp143
    tmp175 = libdevice.rsqrt(tmp174)
    tmp176 = tmp172 * tmp175
    tmp177 = tmp176 * tmp147
    tmp178 = tmp177 + tmp149
    tmp179 = tmp8 - tmp63
    tmp180 = (tmp69 / tmp141)
    tmp181 = tmp180 + tmp143
    tmp182 = libdevice.rsqrt(tmp181)
    tmp183 = tmp179 * tmp182
    tmp184 = tmp183 * tmp147
    tmp185 = tmp184 + tmp149
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp17, r0_mask & xmask)
    tl.store(in_out_ptr2 + (r0_1 + 768*x0), tmp26, r0_mask & xmask)
    tl.store(in_out_ptr3 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
    tl.store(in_out_ptr4 + (r0_1 + 768*x0), tmp44, r0_mask & xmask)
    tl.store(in_out_ptr5 + (r0_1 + 768*x0), tmp53, r0_mask & xmask)
    tl.store(out_ptr12 + (r0_1 + 768*x0), tmp150, r0_mask & xmask)
    tl.store(out_ptr13 + (r0_1 + 768*x0), tmp157, r0_mask & xmask)
    tl.store(out_ptr14 + (r0_1 + 768*x0), tmp164, r0_mask & xmask)
    tl.store(out_ptr15 + (r0_1 + 768*x0), tmp171, r0_mask & xmask)
    tl.store(out_ptr16 + (r0_1 + 768*x0), tmp178, r0_mask & xmask)
    tl.store(out_ptr17 + (r0_1 + 768*x0), tmp185, r0_mask & xmask)
