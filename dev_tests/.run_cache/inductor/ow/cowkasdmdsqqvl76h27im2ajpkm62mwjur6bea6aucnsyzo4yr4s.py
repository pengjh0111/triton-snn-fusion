# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/an/canm7yhs334lzde2lc4jbyyhixl2gze2kdmsoy4gx2wfigpydtb7.py
# Topologically Sorted Source Nodes: [getitem, input_1], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem => select
#   input_1 => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %convolution : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_select_0 = async_compile.triton('triton_poi_fused_convolution_select_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/hk/chkmu5eak4amttfsu5b4ywrr4z4dsvbpr5eol6n63tbas5vnhnjk.py
# Topologically Sorted Source Nodes: [getitem, input_1], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem => select
#   input_1 => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[64, 3, 11, 11][363, 121, 11, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %select : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %convolution : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_select_1 = async_compile.triton('triton_poi_fused_convolution_select_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 185856, 'x': 92928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 121*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 363*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/6y/c6y6fxttpvgmdmgef26uddooqel6tg2zs3msmecmoz4wu7kfbo3y.py
# Topologically Sorted Source Nodes: [input_2, zeros_like, lif_forward_state_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_2 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   lif_forward_state_default => lif_forward_state
#   zeros_like => full_default
# Graph fragment:
#   %convolution : Tensor "f32[4, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %full_default : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 64, 55, 55], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_1, %full_default, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf3
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 3098624, 'x': 6195200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 193600*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/pc/cpc2abo6cprfgskvaedee3bzwekxijebuh3myrfn2idbzcp2em5g.py
# Topologically Sorted Source Nodes: [input_2, zeros_like, lif_forward_state_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_2 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   lif_forward_state_default => lif_forward_state
#   zeros_like => full_default
# Graph fragment:
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %full_default : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 64, 55, 55], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_1, %full_default, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf4
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6195200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/ts/cts7offrzdkqawew5ib5svhppsopfdo4vvnx4hqsxhg6r2jifwqp.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => _low_memory_max_pool_with_offsets, getitem_2
# Graph fragment:
#   %getitem : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = PlaceHolder[target=getitem]
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%getitem, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_2 : Tensor "f32[4, 64, 27, 27][46656, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
#   return %getitem_2
triton_poi_fused_max_pool2d_with_indices_4 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1492992, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 729
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 27)
    x3 = xindex // 27
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (55 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (56 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (57 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (110 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (111 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (112 + 2*x2 + 110*x3 + 3025*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (y0 + 64*x5 + 46656*y1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/di/cdig7fctjidrmief22ezddikcud66cryt55arya66amvtezeqgx4.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convolution_1
# Graph fragment:
#   %arg6_1 : Tensor "f32[192, 64, 5, 5][1600, 25, 5, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_1 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg6_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf9
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2457600, 'x': 1228800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 1600*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/hq/chqe66r44y7jy7rcoudfcwbg7j2wtdxj5idrm5ce2pcrrvlpbnrx.py
# Topologically Sorted Source Nodes: [input_5, zeros_like_1, lif_forward_state_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_5 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   lif_forward_state_default_1 => lif_forward_state_1
#   zeros_like_1 => full_default_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[4, 192, 27, 27][139968, 1, 5184, 192]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %unsqueeze_8 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %full_default_1 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 192, 27, 27], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_1 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_3, %full_default_1, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf11
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2242560, 'x': 4478976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 729
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 139968*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 729*y3), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/l5/cl53mjeku5ol33ofzbbaev66ghmkugr7tjz4slteyaidu7wx5ic4.py
# Topologically Sorted Source Nodes: [input_5, zeros_like_1, lif_forward_state_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_5 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   lif_forward_state_default_1 => lif_forward_state_1
#   zeros_like_1 => full_default_1
# Graph fragment:
#   %unsqueeze_8 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %full_default_1 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 192, 27, 27], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_1 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_3, %full_default_1, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf12
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4478976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 559872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/y5/cy5hfr7tugvi5om5jd2n5okthdsb54pdpzi7yflthpabgv5j4cfg.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_6 => _low_memory_max_pool_with_offsets_1, getitem_6
# Graph fragment:
#   %getitem_4 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%getitem_4, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_6 : Tensor "f32[4, 192, 13, 13][32448, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_1, 0), kwargs = {})
#   return %getitem_6
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1038336, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 13)
    x3 = xindex // 13
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (27 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (29 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (54 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (55 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (56 + 2*x2 + 54*x3 + 729*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (y0 + 192*x5 + 32448*y1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/kg/ckg3npdsvywphrr2ragvzhwsxgpuzincf655k4ggkgitiqaqjuen.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_7 => convolution_2
# Graph fragment:
#   %arg11_1 : Tensor "f32[384, 192, 3, 3][1728, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution_2 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf17
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 5308416, 'x': 2654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/h2/ch2z56xasmhymaohp2wg7tlytoa7umrjl5qwiziyqgjjvbjn4bu5.py
# Topologically Sorted Source Nodes: [input_8, zeros_like_2, lif_forward_state_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_8 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   lif_forward_state_default_2 => lif_forward_state_2
#   zeros_like_2 => full_default_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[4, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg12_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %unsqueeze_16 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %full_default_2 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 384, 13, 13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_2 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_5, %full_default_2, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf19
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1044480, 'x': 2076672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 384*x2 + 64896*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 169*y3), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/zv/czvf4nuviphwsnjr2mdtvkszxyofdoyssdype3ung4xu2wwiohqt.py
# Topologically Sorted Source Nodes: [input_8, zeros_like_2, lif_forward_state_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_8 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   lif_forward_state_default_2 => lif_forward_state_2
#   zeros_like_2 => full_default_2
# Graph fragment:
#   %unsqueeze_16 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[384, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[384, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %full_default_2 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 384, 13, 13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_2 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_5, %full_default_2, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf20
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2076672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 259584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/m2/cm2mbr3q2nplecld2ndbwuoxpahkb245jxqrqh7iwvlg5cbdd5jt.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_9 => convolution_3
# Graph fragment:
#   %getitem_8 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0" = PlaceHolder[target=getitem_8]
#   %convolution_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf24
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2076672, 'x': 1038336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 169*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 64896*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/gk/cgkikf3gtpsti722h3ubvn4l7riw6nrhfomxbgbc77arjm23dpjg.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_9 => convolution_3
# Graph fragment:
#   %arg16_1 : Tensor "f32[256, 384, 3, 3][3456, 9, 3, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf25
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 7077888, 'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/gi/cgiaceiuwbuslx5o3ygmqkj2q7zu6ohrswgcamxdvc4cro372a7w.py
# Topologically Sorted Source Nodes: [input_10, zeros_like_3, lif_forward_state_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_10 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   lif_forward_state_default_3 => lif_forward_state_3
#   zeros_like_3 => full_default_3
# Graph fragment:
#   %convolution_3 : Tensor "f32[4, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = PlaceHolder[target=convolution_3]
#   %arg17_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %unsqueeze_24 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %full_default_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 13, 13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_3 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_7, %full_default_3, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf27
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 696320, 'x': 1384448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + 169*y3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/zm/czm5ffbfpm6dw2v73pf55z3ypgzydo6wioh7t65qbmal3ts2cpnk.py
# Topologically Sorted Source Nodes: [input_10, zeros_like_3, lif_forward_state_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   input_10 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   lif_forward_state_default_3 => lif_forward_state_3
#   zeros_like_3 => full_default_3
# Graph fragment:
#   %unsqueeze_24 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %full_default_3 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 13, 13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_3 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%add_7, %full_default_3, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf28
triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1384448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 173056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/6n/c6nsxmkdfnxhnbn5e6ghln5sbo7tzyf6tzcwq5mlf4welclkljq3.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_11 => convolution_4
# Graph fragment:
#   %getitem_10 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0" = PlaceHolder[target=getitem_10]
#   %convolution_4 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf32
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1384448, 'x': 692224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 169*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 43264*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/x2/cx27v4vwvnm3d536sk5mueyvuhhsemmevffxvddner25mvx3pn77.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_11 => convolution_4
# Graph fragment:
#   %arg21_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convolution_4 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf33
triton_poi_fused_convolution_17 = async_compile.triton('triton_poi_fused_convolution_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/p5/cp5d4qv65vheuurezhoxtgfsm3u7c6ybrknra5bwc5iz42nd6cgd.py
# Topologically Sorted Source Nodes: [input_13, x], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   input_13 => _low_memory_max_pool_with_offsets_2, getitem_14
#   x => _adaptive_avg_pool2d
# Graph fragment:
#   %getitem_12 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %getitem_14 : Tensor "f32[4, 256, 6, 6][9216, 1, 1536, 256]cuda:0" = PlaceHolder[target=getitem_14]
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%getitem_12, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_14 : Tensor "f32[4, 256, 6, 6][9216, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_2, 0), kwargs = {})
#   %_adaptive_avg_pool2d : Tensor "f32[4, 256, 6, 6][9216, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%getitem_14, [6, 6]), kwargs = {})
#   return %getitem_14,%_adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x5 = xindex // 36
    x2 = ((xindex // 36) % 256)
    x3 = xindex // 9216
    x4 = (xindex % 36)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (27 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + 2*x0 + 26*x1 + 169*x5), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr1 + (x6), tmp16, None)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/7g/c7gcejotyy6pl4yn2euytxjihhelglqd55djq6lsparya64tpike.py
# Topologically Sorted Source Nodes: [zeros_like_5, lif_forward_state_default_5], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   lif_forward_state_default_5 => lif_forward_state_5
#   zeros_like_5 => full_default_5
# Graph fragment:
#   %full_default_5 : Tensor "f32[4, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4096], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_5 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%mm, %full_default_5, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf43
triton_poi_fused_lif_forward_state_zeros_like_19 = async_compile.triton('triton_poi_fused_lif_forward_state_zeros_like_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_lif_forward_state_zeros_like_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_lif_forward_state_zeros_like_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/hn/chnhjmsukybuiriqgdf3pt7gkdrzmasnzqdfrd2prnkkqggljb4b.py
# Topologically Sorted Source Nodes: [zeros_like_7, lif_forward_state_default_7], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
# Source node to ATen node mapping:
#   lif_forward_state_default_7 => lif_forward_state_7
#   zeros_like_7 => full_default_7
# Graph fragment:
#   %full_default_7 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lif_forward_state_7 : [num_users=2] = call_function[target=torch.ops.snn_custom.lif_forward_state.default](args = (%mm_2, %full_default_7, 1.0, 0.0, 2.0, False), kwargs = {})
#   return %buf53
triton_poi_fused_lif_forward_state_zeros_like_20 = async_compile.triton('triton_poi_fused_lif_forward_state_zeros_like_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_lif_forward_state_zeros_like_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_lif_forward_state_zeros_like_20(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/pu/cpum742mfqvet2cfau4sodszv2ti642hf5ydxwnrgi6c7tlifuky.py
# Topologically Sorted Source Nodes: [getitem_17, input_19], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_17 => select_1
#   input_19 => convolution_5
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_1 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %convolution_5 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_1, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf57
triton_poi_fused_convolution_select_21 = async_compile.triton('triton_poi_fused_convolution_select_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (602112 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/7y/c7ydfyrsdvmqtsix6owfctpl6q4ec67ebnac5s32dt3dtzjx2va4.py
# Topologically Sorted Source Nodes: [getitem_17, input_19, getitem_34, input_37], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_17 => select_1
#   getitem_34 => select_2
#   input_19 => convolution_5
#   input_37 => convolution_10
# Graph fragment:
#   %arg1_1 : Tensor "f32[64, 3, 11, 11][363, 121, 11, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %select_1 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %convolution_5 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_1, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %select_2 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 2), kwargs = {})
#   %convolution_10 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_2, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf58,%buf107
triton_poi_fused_convolution_select_22 = async_compile.triton('triton_poi_fused_convolution_select_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 371712, 'x': 92928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_22(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 121*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 363*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 3*x2 + 363*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/xr/cxrhkpbcc6h33hokz3bnxrvr4bozydly2sjhskzwcv4c6avmeatt.py
# Topologically Sorted Source Nodes: [input_22, input_40], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_22 => convolution_6
#   input_40 => convolution_11
# Graph fragment:
#   %arg6_1 : Tensor "f32[192, 64, 5, 5][1600, 25, 5, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_6 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_24, %arg6_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_46, %arg6_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf65,%buf114
triton_poi_fused_convolution_23 = async_compile.triton('triton_poi_fused_convolution_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4915200, 'x': 1228800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_23(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 1600*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 1600*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/fr/cfrwntf3cvgdnradjqeuhxzym4q3cblrj7ozsfyvm2ebjdfzn5uc.py
# Topologically Sorted Source Nodes: [input_25, input_43], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_25 => convolution_7
#   input_43 => convolution_12
# Graph fragment:
#   %arg11_1 : Tensor "f32[384, 192, 3, 3][1728, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution_7 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_28, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_12 : Tensor "f32[4, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_50, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf72,%buf121
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 10616832, 'x': 2654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/fx/cfxm3hteendrl42agtckhijpwbrazw56jwyt3ehahuxqyalqcdat.py
# Topologically Sorted Source Nodes: [input_27, input_45], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_27 => convolution_8
#   input_45 => convolution_13
# Graph fragment:
#   %arg16_1 : Tensor "f32[256, 384, 3, 3][3456, 9, 3, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution_8 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_30, %arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_52, %arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf79,%buf128
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 14155776, 'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/xk/cxkancmxrzts7eadoaenycoyj3acuhlzixm23ohwe3w4wetgptzn.py
# Topologically Sorted Source Nodes: [input_29, input_47], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_29 => convolution_9
#   input_47 => convolution_14
# Graph fragment:
#   %arg21_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convolution_9 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_32, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_14 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_54, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf86,%buf135
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9437184, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/w4/cw4ei3mdipedz3x2nmbiybysoytjotaxkkqm57yfe5ptgqmqxi6o.py
# Topologically Sorted Source Nodes: [getitem_34, input_37], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_34 => select_2
#   input_37 => convolution_10
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_2 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 2), kwargs = {})
#   %convolution_10 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_2, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf106
triton_poi_fused_convolution_select_27 = async_compile.triton('triton_poi_fused_convolution_select_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (1204224 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/gi/cgibhkq2qff4u5f72wjr4dqgaud4jmhjglvuwzkrypgrpfs7czpn.py
# Topologically Sorted Source Nodes: [getitem_51, input_55], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_51 => select_3
#   input_55 => convolution_15
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_3 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 3), kwargs = {})
#   %convolution_15 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_3, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf155
triton_poi_fused_convolution_select_28 = async_compile.triton('triton_poi_fused_convolution_select_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (1806336 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/i2/ci22a4ielctsl2u2z6oyb7as7k4wyatkdfa3n3phux5fxsbpbngw.py
# Topologically Sorted Source Nodes: [getitem_68, input_73], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_68 => select_4
#   input_73 => convolution_20
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_4 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 4), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_4, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf204
triton_poi_fused_convolution_select_29 = async_compile.triton('triton_poi_fused_convolution_select_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (2408448 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/uy/cuy4e56xuuuqr6lnuyy4x4bt5gt3xuj5233hkhi7aslsmysxyvil.py
# Topologically Sorted Source Nodes: [getitem_85, input_91], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_85 => select_5
#   input_91 => convolution_25
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_5 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 5), kwargs = {})
#   %convolution_25 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_5, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf253
triton_poi_fused_convolution_select_30 = async_compile.triton('triton_poi_fused_convolution_select_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (3010560 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/ji/cjitisencte6kcoypzhib74xoh7dm2hokvtchigciyuj7gxfcqcg.py
# Topologically Sorted Source Nodes: [getitem_102, input_109], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_102 => select_6
#   input_109 => convolution_30
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_6 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 6), kwargs = {})
#   %convolution_30 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_6, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf302
triton_poi_fused_convolution_select_31 = async_compile.triton('triton_poi_fused_convolution_select_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (3612672 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/fq/cfq37vvj5iwblx76a65vnxgjojnaq6hmdv4o6nfjef4cstymdopr.py
# Topologically Sorted Source Nodes: [getitem_119, input_127], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_119 => select_7
#   input_127 => convolution_35
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_7 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 7), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_7, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf351
triton_poi_fused_convolution_select_32 = async_compile.triton('triton_poi_fused_convolution_select_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (4214784 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/ds/cdsjlcvrewyzsywo3efakaiuxhgp7rw643dgt7c3xsspfwx6ynb4.py
# Topologically Sorted Source Nodes: [getitem_136, input_145], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_136 => select_8
#   input_145 => convolution_40
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_8 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 8), kwargs = {})
#   %convolution_40 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_8, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf400
triton_poi_fused_convolution_select_33 = async_compile.triton('triton_poi_fused_convolution_select_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_33(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (4816896 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/nj/cnjqf6appuumdpz4mviqfprs2gwvojhku3drdho7id7l23wybrzz.py
# Topologically Sorted Source Nodes: [getitem_153, input_163], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_153 => select_9
#   input_163 => convolution_45
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_9 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 9), kwargs = {})
#   %convolution_45 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_9, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf449
triton_poi_fused_convolution_select_34 = async_compile.triton('triton_poi_fused_convolution_select_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (5419008 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/zc/czckpzyqwm2ejsnvkosijijrvw7hrlxx7wevobnucomeuoctuakv.py
# Topologically Sorted Source Nodes: [getitem_170, input_181], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_170 => select_10
#   input_181 => convolution_50
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_10 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 10), kwargs = {})
#   %convolution_50 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_10, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf498
triton_poi_fused_convolution_select_35 = async_compile.triton('triton_poi_fused_convolution_select_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (6021120 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/si/csijnjtkx35lauwd6rebwu7l4sbbxhzlebbkm7b3kxa7mdnl5ief.py
# Topologically Sorted Source Nodes: [getitem_187, input_199], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_187 => select_11
#   input_199 => convolution_55
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_11 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 11), kwargs = {})
#   %convolution_55 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_11, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf547
triton_poi_fused_convolution_select_36 = async_compile.triton('triton_poi_fused_convolution_select_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (6623232 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/l7/cl75xzmbxwfidqypaulyaoe46ojgzirohi53cduhrpgt6suhfi7w.py
# Topologically Sorted Source Nodes: [getitem_204, input_217], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_204 => select_12
#   input_217 => convolution_60
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_12 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 12), kwargs = {})
#   %convolution_60 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_12, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf596
triton_poi_fused_convolution_select_37 = async_compile.triton('triton_poi_fused_convolution_select_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (7225344 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/z2/cz2tnhi34w2jv2pfwttxyjuirk3svbmzfqv5vstc6xmdgs2dt4zr.py
# Topologically Sorted Source Nodes: [getitem_221, input_235], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_221 => select_13
#   input_235 => convolution_65
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_13 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 13), kwargs = {})
#   %convolution_65 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_13, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf645
triton_poi_fused_convolution_select_38 = async_compile.triton('triton_poi_fused_convolution_select_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (7827456 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/n6/cn6o2vmfxgf5roo7z4glzwd25ngberfz5ep2qk4czv6b65rthb3m.py
# Topologically Sorted Source Nodes: [getitem_238, input_253], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_238 => select_14
#   input_253 => convolution_70
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_14 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 14), kwargs = {})
#   %convolution_70 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_14, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf694
triton_poi_fused_convolution_select_39 = async_compile.triton('triton_poi_fused_convolution_select_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (8429568 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/4k/c4kqywua4gwng7qr75hfrxirklhlhgyjk5fhnzrclrqtd2l6b4co.py
# Topologically Sorted Source Nodes: [getitem_255, input_271], Original ATen: [aten.select, aten.convolution]
# Source node to ATen node mapping:
#   getitem_255 => select_15
#   input_271 => convolution_75
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 4, 3, 224, 224][602112, 150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_15 : Tensor "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 15), kwargs = {})
#   %convolution_75 : Tensor "f32[4, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_15, %arg1_1, None, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf743
triton_poi_fused_convolution_select_40 = async_compile.triton('triton_poi_fused_convolution_select_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_select_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4816896, 'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_select_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (9031680 + x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/6l/c6latyeougme3lln6ubfzvl4klxvfzqn3v6g2chcfrvcmo6ievw6.py
# Topologically Sorted Source Nodes: [input_283, x_30], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   input_283 => _low_memory_max_pool_with_offsets_47
#   x_30 => _adaptive_avg_pool2d_15
# Graph fragment:
#   %getitem_342 : Tensor "f32[4, 256, 13, 13][43264, 169, 13, 1]cuda:0" = PlaceHolder[target=getitem_342]
#   %getitem_344 : Tensor "f32[4, 256, 6, 6][9216, 36, 6, 1]cuda:0" = PlaceHolder[target=getitem_344]
#   %_low_memory_max_pool_with_offsets_47 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%getitem_342, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_adaptive_avg_pool2d_15 : Tensor "f32[4, 256, 6, 6][9216, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%getitem_344, [6, 6]), kwargs = {})
#   return %getitem_344,%_adaptive_avg_pool2d_15
triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_41 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_41(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x2 = xindex // 36
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (27 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + 2*x0 + 26*x1 + 169*x2), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor/jb/cjb2fmecjnextsyx5auu2fmnefhfnb3t23nnonv6s45jrkcx2hy7.py
# Topologically Sorted Source Nodes: [out_spikes_counter, out_spikes_counter_1, out_spikes_counter_2, out_spikes_counter_3, out_spikes_counter_4, out_spikes_counter_5, out_spikes_counter_6, out_spikes_counter_7, out_spikes_counter_8, out_spikes_counter_9, out_spikes_counter_10, out_spikes_counter_11, out_spikes_counter_12, out_spikes_counter_13, out_spikes_counter_14, out_spikes_counter_15, truediv], Original ATen: [aten.add, aten.div]
# Source node to ATen node mapping:
#   out_spikes_counter => add_10
#   out_spikes_counter_1 => add_21
#   out_spikes_counter_10 => add_120
#   out_spikes_counter_11 => add_131
#   out_spikes_counter_12 => add_142
#   out_spikes_counter_13 => add_153
#   out_spikes_counter_14 => add_164
#   out_spikes_counter_15 => add_175
#   out_spikes_counter_2 => add_32
#   out_spikes_counter_3 => add_43
#   out_spikes_counter_4 => add_54
#   out_spikes_counter_5 => add_65
#   out_spikes_counter_6 => add_76
#   out_spikes_counter_7 => add_87
#   out_spikes_counter_8 => add_98
#   out_spikes_counter_9 => add_109
#   truediv => div
# Graph fragment:
#   %getitem_20 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_20]
#   %getitem_42 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_42]
#   %getitem_64 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_64]
#   %getitem_86 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_86]
#   %getitem_108 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_108]
#   %getitem_130 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_130]
#   %getitem_152 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_152]
#   %getitem_174 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_174]
#   %getitem_196 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_196]
#   %add_98 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=add_98]
#   %getitem_218 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_218]
#   %getitem_240 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_240]
#   %getitem_262 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_262]
#   %getitem_284 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_284]
#   %getitem_306 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_306]
#   %getitem_328 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_328]
#   %getitem_350 : Tensor "f32[4, 10][10, 1]cuda:0" = PlaceHolder[target=getitem_350]
#   %add_10 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 0), kwargs = {})
#   %add_21 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %getitem_42), kwargs = {})
#   %add_32 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %getitem_64), kwargs = {})
#   %add_43 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %getitem_86), kwargs = {})
#   %add_54 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %getitem_108), kwargs = {})
#   %add_65 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %getitem_130), kwargs = {})
#   %add_76 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %getitem_152), kwargs = {})
#   %add_87 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %getitem_174), kwargs = {})
#   %add_98 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %getitem_196), kwargs = {})
#   %add_109 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_98, %getitem_218), kwargs = {})
#   %add_120 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %getitem_240), kwargs = {})
#   %add_131 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_120, %getitem_262), kwargs = {})
#   %add_142 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %getitem_284), kwargs = {})
#   %add_153 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_142, %getitem_306), kwargs = {})
#   %add_164 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %getitem_328), kwargs = {})
#   %add_175 : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %getitem_350), kwargs = {})
#   %div : Tensor "f32[4, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_175, 16), kwargs = {})
#   return %add_98,%div
triton_poi_fused_add_div_42 = async_compile.triton('triton_poi_fused_add_div_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 16, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0AEAF87B22C450DA0ABDB10B53A6B2D367C26F52B4066683B64E07FA0250A537', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask)
    tmp7 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    tmp19 = tl.load(in_ptr8 + (x0), xmask)
    tmp21 = tl.load(in_ptr9 + (x0), xmask)
    tmp23 = tl.load(in_ptr10 + (x0), xmask)
    tmp25 = tl.load(in_ptr11 + (x0), xmask)
    tmp27 = tl.load(in_ptr12 + (x0), xmask)
    tmp29 = tl.load(in_ptr13 + (x0), xmask)
    tmp31 = tl.load(in_ptr14 + (x0), xmask)
    tmp1 = tl.full([1], 0.0, tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full([1], 0.0625, tl.float32)
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16, 4, 3, 224, 224), (602112, 150528, 50176, 224, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_misaligned(arg0_1)
            buf0 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem, input_1], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_0.run(arg0_1, buf0, 12, 50176, stream=raw_stream0)
            assert_size_stride(arg1_1, (64, 3, 11, 11), (363, 121, 11, 1))
            arg1_1 = copy_misaligned(arg1_1)
            buf1 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem, input_1], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_1.run(arg1_1, buf1, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem, input_1], Original ATen: [aten.select, aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg2_1, (64, ), (1, ))
            assert_size_stride(arg3_1, (64, ), (1, ))
            assert_size_stride(arg4_1, (64, ), (1, ))
            assert_size_stride(arg5_1, (64, ), (1, ))
            arg2_1 = copy_misaligned(arg2_1)
            arg3_1 = copy_misaligned(arg3_1)
            arg4_1 = copy_misaligned(arg4_1)
            arg5_1 = copy_misaligned(arg5_1)
            buf3 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, zeros_like, lif_forward_state_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf2, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 256, 3025, stream=raw_stream0)
            buf4 = reinterpret_tensor(buf2, (4, 64, 55, 55), (193600, 3025, 55, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [input_2, zeros_like, lif_forward_state_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_3.run(buf4, 774400, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_2, zeros_like, lif_forward_state_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            buf5 = torch.ops.snn_custom.lif_forward_state.default(buf3, buf4, 1.0, 0.0, 2.0, False)
            del buf3
            del buf4
            buf6 = buf5[0]
            assert_size_stride(buf6, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf6, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf7 = buf5[1]
            assert_size_stride(buf7, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf7, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf5
            buf8 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf6, buf8, 256, 729, stream=raw_stream0)
            assert_size_stride(arg6_1, (192, 64, 5, 5), (1600, 25, 5, 1))
            arg6_1 = copy_misaligned(arg6_1)
            buf9 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_5.run(arg6_1, buf9, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf10, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg7_1, (192, ), (1, ))
            assert_size_stride(arg8_1, (192, ), (1, ))
            assert_size_stride(arg9_1, (192, ), (1, ))
            assert_size_stride(arg10_1, (192, ), (1, ))
            arg7_1 = copy_misaligned(arg7_1)
            arg8_1 = copy_misaligned(arg8_1)
            arg9_1 = copy_misaligned(arg9_1)
            arg10_1 = copy_misaligned(arg10_1)
            buf11 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_5, zeros_like_1, lif_forward_state_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf10, arg7_1, arg8_1, arg9_1, arg10_1, buf11, 768, 729, stream=raw_stream0)
            buf12 = reinterpret_tensor(buf10, (4, 192, 27, 27), (139968, 729, 27, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_5, zeros_like_1, lif_forward_state_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_7.run(buf12, 559872, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_5, zeros_like_1, lif_forward_state_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            buf13 = torch.ops.snn_custom.lif_forward_state.default(buf11, buf12, 1.0, 0.0, 2.0, False)
            del buf11
            del buf12
            buf14 = buf13[0]
            assert_size_stride(buf14, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf14, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf15 = buf13[1]
            assert_size_stride(buf15, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf15, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf13
            buf16 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf14, buf16, 768, 169, stream=raw_stream0)
            assert_size_stride(arg11_1, (384, 192, 3, 3), (1728, 9, 3, 1))
            arg11_1 = copy_misaligned(arg11_1)
            buf17 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_9.run(arg11_1, buf17, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
            buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg12_1, (384, ), (1, ))
            assert_size_stride(arg13_1, (384, ), (1, ))
            assert_size_stride(arg14_1, (384, ), (1, ))
            assert_size_stride(arg15_1, (384, ), (1, ))
            arg12_1 = copy_misaligned(arg12_1)
            arg13_1 = copy_misaligned(arg13_1)
            arg14_1 = copy_misaligned(arg14_1)
            arg15_1 = copy_misaligned(arg15_1)
            buf19 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_8, zeros_like_2, lif_forward_state_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf18, arg12_1, arg13_1, arg14_1, arg15_1, buf19, 1536, 169, stream=raw_stream0)
            buf20 = reinterpret_tensor(buf18, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_8, zeros_like_2, lif_forward_state_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_11.run(buf20, 259584, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_8, zeros_like_2, lif_forward_state_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            buf21 = torch.ops.snn_custom.lif_forward_state.default(buf19, buf20, 1.0, 0.0, 2.0, False)
            del buf19
            buf22 = buf21[0]
            assert_size_stride(buf22, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf22, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf23 = buf21[1]
            assert_size_stride(buf23, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf23, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf21
            buf24 = reinterpret_tensor(buf20, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf22, buf24, 1536, 169, stream=raw_stream0)
            del buf22
            assert_size_stride(arg16_1, (256, 384, 3, 3), (3456, 9, 3, 1))
            arg16_1 = copy_misaligned(arg16_1)
            buf25 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_13.run(arg16_1, buf25, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
            buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf25
            assert_size_stride(arg17_1, (256, ), (1, ))
            assert_size_stride(arg18_1, (256, ), (1, ))
            assert_size_stride(arg19_1, (256, ), (1, ))
            assert_size_stride(arg20_1, (256, ), (1, ))
            arg17_1 = copy_misaligned(arg17_1)
            arg18_1 = copy_misaligned(arg18_1)
            arg19_1 = copy_misaligned(arg19_1)
            arg20_1 = copy_misaligned(arg20_1)
            buf27 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, zeros_like_3, lif_forward_state_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf26, arg17_1, arg18_1, arg19_1, arg20_1, buf27, 1024, 169, stream=raw_stream0)
            buf28 = reinterpret_tensor(buf26, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [input_10, zeros_like_3, lif_forward_state_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15.run(buf28, 173056, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_10, zeros_like_3, lif_forward_state_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            buf29 = torch.ops.snn_custom.lif_forward_state.default(buf27, buf28, 1.0, 0.0, 2.0, False)
            del buf27
            buf30 = buf29[0]
            assert_size_stride(buf30, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf30, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf31 = buf29[1]
            assert_size_stride(buf31, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf31, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf29
            buf32 = reinterpret_tensor(buf28, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf30, buf32, 1024, 169, stream=raw_stream0)
            del buf30
            assert_size_stride(arg21_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            arg21_1 = copy_misaligned(arg21_1)
            buf33 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_17.run(arg21_1, buf33, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf33
            assert_size_stride(arg22_1, (256, ), (1, ))
            assert_size_stride(arg23_1, (256, ), (1, ))
            assert_size_stride(arg24_1, (256, ), (1, ))
            assert_size_stride(arg25_1, (256, ), (1, ))
            arg22_1 = copy_misaligned(arg22_1)
            arg23_1 = copy_misaligned(arg23_1)
            arg24_1 = copy_misaligned(arg24_1)
            arg25_1 = copy_misaligned(arg25_1)
            buf35 = reinterpret_tensor(buf32, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_12, zeros_like_4, lif_forward_state_default_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf34, arg22_1, arg23_1, arg24_1, arg25_1, buf35, 1024, 169, stream=raw_stream0)
            buf36 = reinterpret_tensor(buf34, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [input_12, zeros_like_4, lif_forward_state_default_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_15.run(buf36, 173056, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_12, zeros_like_4, lif_forward_state_default_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.zeros_like, snn_custom.lif_forward_state]
            buf37 = torch.ops.snn_custom.lif_forward_state.default(buf35, buf36, 1.0, 0.0, 2.0, False)
            del buf35
            del buf36
            buf38 = buf37[0]
            assert_size_stride(buf38, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf38, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf39 = buf37[1]
            assert_size_stride(buf39, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf39, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf37
            buf41 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_13, x], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf38, buf41, 36864, stream=raw_stream0)
            assert_size_stride(arg26_1, (4096, 9216), (9216, 1))
            arg26_1 = copy_misaligned(arg26_1)
            buf42 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, input_15], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf41, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf42)
            buf43 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like_5, lif_forward_state_default_5], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_lif_forward_state_zeros_like_19.run(buf43, 16384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [zeros_like_5, lif_forward_state_default_5], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            buf44 = torch.ops.snn_custom.lif_forward_state.default(buf42, buf43, 1.0, 0.0, 2.0, False)
            del buf42
            buf45 = buf44[0]
            assert_size_stride(buf45, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf45, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf46 = buf44[1]
            assert_size_stride(buf46, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf46, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf44
            assert_size_stride(arg27_1, (4096, 4096), (4096, 1))
            arg27_1 = copy_misaligned(arg27_1)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf45, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf47)
            buf48 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [zeros_like_6, lif_forward_state_default_6], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_lif_forward_state_zeros_like_19.run(buf48, 16384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [zeros_like_6, lif_forward_state_default_6], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            buf49 = torch.ops.snn_custom.lif_forward_state.default(buf47, buf48, 1.0, 0.0, 2.0, False)
            del buf47
            del buf48
            buf50 = buf49[0]
            assert_size_stride(buf50, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf50, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf51 = buf49[1]
            assert_size_stride(buf51, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf51, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf49
            assert_size_stride(arg28_1, (10, 4096), (4096, 1))
            arg28_1 = copy_misaligned(arg28_1)
            buf52 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf50, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf52)
            buf53 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like_7, lif_forward_state_default_7], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_lif_forward_state_zeros_like_20.run(buf53, 40, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [zeros_like_7, lif_forward_state_default_7], Original ATen: [aten.zeros_like, snn_custom.lif_forward_state]
            buf54 = torch.ops.snn_custom.lif_forward_state.default(buf52, buf53, 1.0, 0.0, 2.0, False)
            del buf52
            buf55 = buf54[0]
            assert_size_stride(buf55, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf55, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf56 = buf54[1]
            assert_size_stride(buf56, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf56, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf54
            buf57 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [getitem_17, input_19], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_21.run(arg0_1, buf57, 12, 50176, stream=raw_stream0)
            buf58 = buf1; del buf1  # reuse
            buf107 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_17, input_19, getitem_34, input_37], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf58, buf107, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_17, input_19], Original ATen: [aten.select, aten.convolution]
            buf59 = extern_kernels.convolution(buf57, buf58, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf59, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf57
            del buf58
            buf60 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [input_20, lif_forward_state_default_8], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf59, arg2_1, arg3_1, arg4_1, arg5_1, buf60, 256, 3025, stream=raw_stream0)
            del buf59
            # Topologically Sorted Source Nodes: [input_20, lif_forward_state_default_8], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf61 = torch.ops.snn_custom.lif_forward_state.default(buf60, buf7, 1.0, 0.0, 2.0, False)
            del buf60
            del buf7
            buf62 = buf61[0]
            assert_size_stride(buf62, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf62, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf63 = buf61[1]
            assert_size_stride(buf63, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf63, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf61
            buf64 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf62, buf64, 256, 729, stream=raw_stream0)
            buf65 = buf9; del buf9  # reuse
            buf114 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_22, input_40], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf65, buf114, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
            buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf66, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf64
            del buf65
            buf67 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [input_23, lif_forward_state_default_9], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf66, arg7_1, arg8_1, arg9_1, arg10_1, buf67, 768, 729, stream=raw_stream0)
            del buf66
            # Topologically Sorted Source Nodes: [input_23, lif_forward_state_default_9], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf68 = torch.ops.snn_custom.lif_forward_state.default(buf67, buf15, 1.0, 0.0, 2.0, False)
            del buf15
            del buf67
            buf69 = buf68[0]
            assert_size_stride(buf69, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf69, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf70 = buf68[1]
            assert_size_stride(buf70, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf70, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf68
            buf71 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf69, buf71, 768, 169, stream=raw_stream0)
            del buf69
            buf72 = buf17; del buf17  # reuse
            buf121 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_25, input_43], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf72, buf121, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
            buf73 = extern_kernels.convolution(buf71, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf73, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf71
            del buf72
            buf74 = reinterpret_tensor(buf24, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_26, lif_forward_state_default_10], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf73, arg12_1, arg13_1, arg14_1, arg15_1, buf74, 1536, 169, stream=raw_stream0)
            del buf73
            # Topologically Sorted Source Nodes: [input_26, lif_forward_state_default_10], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf75 = torch.ops.snn_custom.lif_forward_state.default(buf74, buf23, 1.0, 0.0, 2.0, False)
            del buf23
            buf76 = buf75[0]
            assert_size_stride(buf76, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf76, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf77 = buf75[1]
            assert_size_stride(buf77, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf77, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf75
            buf78 = reinterpret_tensor(buf74, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf76, buf78, 1536, 169, stream=raw_stream0)
            del buf76
            buf79 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf128 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_27, input_45], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf79, buf128, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
            buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf80, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf78
            del buf79
            buf81 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [input_28, lif_forward_state_default_11], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf80, arg17_1, arg18_1, arg19_1, arg20_1, buf81, 1024, 169, stream=raw_stream0)
            del buf80
            # Topologically Sorted Source Nodes: [input_28, lif_forward_state_default_11], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf82 = torch.ops.snn_custom.lif_forward_state.default(buf81, buf31, 1.0, 0.0, 2.0, False)
            del buf31
            buf83 = buf82[0]
            assert_size_stride(buf83, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf83, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf84 = buf82[1]
            assert_size_stride(buf84, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf84, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf82
            buf85 = reinterpret_tensor(buf81, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf83, buf85, 1024, 169, stream=raw_stream0)
            del buf83
            buf86 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            buf135 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_29, input_47], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf86, buf135, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
            buf87 = extern_kernels.convolution(buf85, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf87, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf86
            buf88 = reinterpret_tensor(buf85, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [input_30, lif_forward_state_default_12], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf87, arg22_1, arg23_1, arg24_1, arg25_1, buf88, 1024, 169, stream=raw_stream0)
            del buf87
            # Topologically Sorted Source Nodes: [input_30, lif_forward_state_default_12], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf89 = torch.ops.snn_custom.lif_forward_state.default(buf88, buf39, 1.0, 0.0, 2.0, False)
            del buf39
            del buf88
            buf90 = buf89[0]
            assert_size_stride(buf90, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf90, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf91 = buf89[1]
            assert_size_stride(buf91, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf91, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf89
            buf93 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [input_31, x_2], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf90, buf93, 36864, stream=raw_stream0)
            del buf90
            buf94 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [x_2, x_3, input_33], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf94)
            del buf93
            # Topologically Sorted Source Nodes: [lif_forward_state_default_13], Original ATen: [snn_custom.lif_forward_state]
            buf95 = torch.ops.snn_custom.lif_forward_state.default(buf94, buf46, 1.0, 0.0, 2.0, False)
            del buf46
            buf96 = buf95[0]
            assert_size_stride(buf96, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf96, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf97 = buf95[1]
            assert_size_stride(buf97, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf97, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf95
            buf98 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf96, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf98)
            del buf96
            # Topologically Sorted Source Nodes: [lif_forward_state_default_14], Original ATen: [snn_custom.lif_forward_state]
            buf99 = torch.ops.snn_custom.lif_forward_state.default(buf98, buf51, 1.0, 0.0, 2.0, False)
            del buf51
            del buf98
            buf100 = buf99[0]
            assert_size_stride(buf100, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf100, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf101 = buf99[1]
            assert_size_stride(buf101, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf101, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf99
            buf102 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf100, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf102)
            del buf100
            # Topologically Sorted Source Nodes: [lif_forward_state_default_15], Original ATen: [snn_custom.lif_forward_state]
            buf103 = torch.ops.snn_custom.lif_forward_state.default(buf102, buf56, 1.0, 0.0, 2.0, False)
            del buf102
            buf104 = buf103[0]
            assert_size_stride(buf104, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf104, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf105 = buf103[1]
            assert_size_stride(buf105, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf105, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf103
            buf106 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_34, input_37], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_27.run(arg0_1, buf106, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_34, input_37], Original ATen: [aten.select, aten.convolution]
            buf108 = extern_kernels.convolution(buf106, buf107, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf108, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf106
            del buf107
            buf109 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [input_38, lif_forward_state_default_16], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf108, arg2_1, arg3_1, arg4_1, arg5_1, buf109, 256, 3025, stream=raw_stream0)
            del buf108
            # Topologically Sorted Source Nodes: [input_38, lif_forward_state_default_16], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf110 = torch.ops.snn_custom.lif_forward_state.default(buf109, buf63, 1.0, 0.0, 2.0, False)
            del buf109
            del buf63
            buf111 = buf110[0]
            assert_size_stride(buf111, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf111, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf112 = buf110[1]
            assert_size_stride(buf112, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf112, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf110
            buf113 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf111, buf113, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
            buf115 = extern_kernels.convolution(buf113, buf114, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf115, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf113
            del buf114
            buf116 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_41, lif_forward_state_default_17], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf115, arg7_1, arg8_1, arg9_1, arg10_1, buf116, 768, 729, stream=raw_stream0)
            del buf115
            # Topologically Sorted Source Nodes: [input_41, lif_forward_state_default_17], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf117 = torch.ops.snn_custom.lif_forward_state.default(buf116, buf70, 1.0, 0.0, 2.0, False)
            del buf116
            del buf70
            buf118 = buf117[0]
            assert_size_stride(buf118, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf118, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf119 = buf117[1]
            assert_size_stride(buf119, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf119, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf117
            buf120 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf118, buf120, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
            buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf122, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf123 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_44, lif_forward_state_default_18], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf122, arg12_1, arg13_1, arg14_1, arg15_1, buf123, 1536, 169, stream=raw_stream0)
            del buf122
            # Topologically Sorted Source Nodes: [input_44, lif_forward_state_default_18], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf124 = torch.ops.snn_custom.lif_forward_state.default(buf123, buf77, 1.0, 0.0, 2.0, False)
            del buf123
            buf125 = buf124[0]
            assert_size_stride(buf125, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf125, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf126 = buf124[1]
            assert_size_stride(buf126, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf126, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf124
            buf127 = reinterpret_tensor(buf77, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf125, buf127, 1536, 169, stream=raw_stream0)
            del buf125
            # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
            buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf129, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf128
            buf130 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_46, lif_forward_state_default_19], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf129, arg17_1, arg18_1, arg19_1, arg20_1, buf130, 1024, 169, stream=raw_stream0)
            del buf129
            # Topologically Sorted Source Nodes: [input_46, lif_forward_state_default_19], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf131 = torch.ops.snn_custom.lif_forward_state.default(buf130, buf84, 1.0, 0.0, 2.0, False)
            del buf130
            buf132 = buf131[0]
            assert_size_stride(buf132, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf132, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf133 = buf131[1]
            assert_size_stride(buf133, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf133, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf131
            buf134 = reinterpret_tensor(buf84, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf132, buf134, 1024, 169, stream=raw_stream0)
            del buf132
            # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
            buf136 = extern_kernels.convolution(buf134, buf135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf136, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf137 = reinterpret_tensor(buf134, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [input_48, lif_forward_state_default_20], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf136, arg22_1, arg23_1, arg24_1, arg25_1, buf137, 1024, 169, stream=raw_stream0)
            del buf136
            # Topologically Sorted Source Nodes: [input_48, lif_forward_state_default_20], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf138 = torch.ops.snn_custom.lif_forward_state.default(buf137, buf91, 1.0, 0.0, 2.0, False)
            del buf137
            del buf91
            buf139 = buf138[0]
            assert_size_stride(buf139, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf139, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf140 = buf138[1]
            assert_size_stride(buf140, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf140, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf138
            buf142 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_49, x_4], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf139, buf142, 36864, stream=raw_stream0)
            buf143 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, input_51], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf142, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf143)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_21], Original ATen: [snn_custom.lif_forward_state]
            buf144 = torch.ops.snn_custom.lif_forward_state.default(buf143, buf97, 1.0, 0.0, 2.0, False)
            del buf143
            buf145 = buf144[0]
            assert_size_stride(buf145, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf145, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf146 = buf144[1]
            assert_size_stride(buf146, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf146, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf144
            buf147 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf145, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf147)
            del buf145
            # Topologically Sorted Source Nodes: [lif_forward_state_default_22], Original ATen: [snn_custom.lif_forward_state]
            buf148 = torch.ops.snn_custom.lif_forward_state.default(buf147, buf101, 1.0, 0.0, 2.0, False)
            del buf101
            del buf147
            buf149 = buf148[0]
            assert_size_stride(buf149, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf149, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf150 = buf148[1]
            assert_size_stride(buf150, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf150, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf148
            buf151 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf149, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf151)
            del buf149
            # Topologically Sorted Source Nodes: [lif_forward_state_default_23], Original ATen: [snn_custom.lif_forward_state]
            buf152 = torch.ops.snn_custom.lif_forward_state.default(buf151, buf105, 1.0, 0.0, 2.0, False)
            del buf105
            buf153 = buf152[0]
            assert_size_stride(buf153, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf153, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf154 = buf152[1]
            assert_size_stride(buf154, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf154, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf152
            buf155 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_51, input_55], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_28.run(arg0_1, buf155, 12, 50176, stream=raw_stream0)
            buf156 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf205 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_51, input_55, getitem_68, input_73], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf156, buf205, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_51, input_55], Original ATen: [aten.select, aten.convolution]
            buf157 = extern_kernels.convolution(buf155, buf156, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf157, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf155
            del buf156
            buf158 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [input_56, lif_forward_state_default_24], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf157, arg2_1, arg3_1, arg4_1, arg5_1, buf158, 256, 3025, stream=raw_stream0)
            del buf157
            # Topologically Sorted Source Nodes: [input_56, lif_forward_state_default_24], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf159 = torch.ops.snn_custom.lif_forward_state.default(buf158, buf112, 1.0, 0.0, 2.0, False)
            del buf112
            del buf158
            buf160 = buf159[0]
            assert_size_stride(buf160, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf160, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf161 = buf159[1]
            assert_size_stride(buf161, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf161, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf159
            buf162 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf160, buf162, 256, 729, stream=raw_stream0)
            del buf160
            buf163 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf212 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_58, input_76], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf163, buf212, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
            buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf164, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf162
            del buf163
            buf165 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [input_59, lif_forward_state_default_25], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf164, arg7_1, arg8_1, arg9_1, arg10_1, buf165, 768, 729, stream=raw_stream0)
            del buf164
            # Topologically Sorted Source Nodes: [input_59, lif_forward_state_default_25], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf166 = torch.ops.snn_custom.lif_forward_state.default(buf165, buf119, 1.0, 0.0, 2.0, False)
            del buf119
            del buf165
            buf167 = buf166[0]
            assert_size_stride(buf167, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf167, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf168 = buf166[1]
            assert_size_stride(buf168, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf168, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf166
            buf169 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf167, buf169, 768, 169, stream=raw_stream0)
            del buf167
            buf170 = buf121; del buf121  # reuse
            buf219 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_61, input_79], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf170, buf219, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
            buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf171, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf169
            del buf170
            buf172 = reinterpret_tensor(buf127, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [input_62, lif_forward_state_default_26], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf171, arg12_1, arg13_1, arg14_1, arg15_1, buf172, 1536, 169, stream=raw_stream0)
            del buf171
            # Topologically Sorted Source Nodes: [input_62, lif_forward_state_default_26], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf173 = torch.ops.snn_custom.lif_forward_state.default(buf172, buf126, 1.0, 0.0, 2.0, False)
            del buf126
            buf174 = buf173[0]
            assert_size_stride(buf174, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf174, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf175 = buf173[1]
            assert_size_stride(buf175, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf175, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf173
            buf176 = reinterpret_tensor(buf172, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf174, buf176, 1536, 169, stream=raw_stream0)
            del buf174
            buf177 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf226 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_63, input_81], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf177, buf226, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
            buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf178, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf176
            del buf177
            buf179 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [input_64, lif_forward_state_default_27], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf178, arg17_1, arg18_1, arg19_1, arg20_1, buf179, 1024, 169, stream=raw_stream0)
            del buf178
            # Topologically Sorted Source Nodes: [input_64, lif_forward_state_default_27], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf180 = torch.ops.snn_custom.lif_forward_state.default(buf179, buf133, 1.0, 0.0, 2.0, False)
            del buf133
            buf181 = buf180[0]
            assert_size_stride(buf181, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf181, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf182 = buf180[1]
            assert_size_stride(buf182, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf182, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf180
            buf183 = reinterpret_tensor(buf179, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf181, buf183, 1024, 169, stream=raw_stream0)
            del buf181
            buf184 = buf135; del buf135  # reuse
            buf233 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_65, input_83], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf184, buf233, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
            buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf185, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf184
            buf186 = reinterpret_tensor(buf183, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [input_66, lif_forward_state_default_28], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf185, arg22_1, arg23_1, arg24_1, arg25_1, buf186, 1024, 169, stream=raw_stream0)
            del buf185
            # Topologically Sorted Source Nodes: [input_66, lif_forward_state_default_28], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf187 = torch.ops.snn_custom.lif_forward_state.default(buf186, buf140, 1.0, 0.0, 2.0, False)
            del buf140
            del buf186
            buf188 = buf187[0]
            assert_size_stride(buf188, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf188, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf189 = buf187[1]
            assert_size_stride(buf189, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf189, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf187
            buf191 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [input_67, x_6], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf188, buf191, 36864, stream=raw_stream0)
            del buf188
            buf192 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, input_69], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf191, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf192)
            del buf191
            # Topologically Sorted Source Nodes: [lif_forward_state_default_29], Original ATen: [snn_custom.lif_forward_state]
            buf193 = torch.ops.snn_custom.lif_forward_state.default(buf192, buf146, 1.0, 0.0, 2.0, False)
            del buf146
            buf194 = buf193[0]
            assert_size_stride(buf194, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf194, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf195 = buf193[1]
            assert_size_stride(buf195, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf195, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf193
            buf196 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf194, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf196)
            del buf194
            # Topologically Sorted Source Nodes: [lif_forward_state_default_30], Original ATen: [snn_custom.lif_forward_state]
            buf197 = torch.ops.snn_custom.lif_forward_state.default(buf196, buf150, 1.0, 0.0, 2.0, False)
            del buf150
            del buf196
            buf198 = buf197[0]
            assert_size_stride(buf198, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf198, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf199 = buf197[1]
            assert_size_stride(buf199, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf199, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf197
            buf200 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf198, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf200)
            del buf198
            # Topologically Sorted Source Nodes: [lif_forward_state_default_31], Original ATen: [snn_custom.lif_forward_state]
            buf201 = torch.ops.snn_custom.lif_forward_state.default(buf200, buf154, 1.0, 0.0, 2.0, False)
            del buf154
            buf202 = buf201[0]
            assert_size_stride(buf202, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf202, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf203 = buf201[1]
            assert_size_stride(buf203, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf203, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf201
            buf204 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_68, input_73], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_29.run(arg0_1, buf204, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_68, input_73], Original ATen: [aten.select, aten.convolution]
            buf206 = extern_kernels.convolution(buf204, buf205, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf206, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf204
            del buf205
            buf207 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_74, lif_forward_state_default_32], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf206, arg2_1, arg3_1, arg4_1, arg5_1, buf207, 256, 3025, stream=raw_stream0)
            del buf206
            # Topologically Sorted Source Nodes: [input_74, lif_forward_state_default_32], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf208 = torch.ops.snn_custom.lif_forward_state.default(buf207, buf161, 1.0, 0.0, 2.0, False)
            del buf161
            del buf207
            buf209 = buf208[0]
            assert_size_stride(buf209, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf209, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf210 = buf208[1]
            assert_size_stride(buf210, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf210, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf208
            buf211 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf209, buf211, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
            buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf213, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf211
            del buf212
            buf214 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_77, lif_forward_state_default_33], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf213, arg7_1, arg8_1, arg9_1, arg10_1, buf214, 768, 729, stream=raw_stream0)
            del buf213
            # Topologically Sorted Source Nodes: [input_77, lif_forward_state_default_33], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf215 = torch.ops.snn_custom.lif_forward_state.default(buf214, buf168, 1.0, 0.0, 2.0, False)
            del buf168
            del buf214
            buf216 = buf215[0]
            assert_size_stride(buf216, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf216, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf217 = buf215[1]
            assert_size_stride(buf217, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf217, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf215
            buf218 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf216, buf218, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
            buf220 = extern_kernels.convolution(buf218, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf220, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf221 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_80, lif_forward_state_default_34], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf220, arg12_1, arg13_1, arg14_1, arg15_1, buf221, 1536, 169, stream=raw_stream0)
            del buf220
            # Topologically Sorted Source Nodes: [input_80, lif_forward_state_default_34], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf222 = torch.ops.snn_custom.lif_forward_state.default(buf221, buf175, 1.0, 0.0, 2.0, False)
            del buf175
            buf223 = buf222[0]
            assert_size_stride(buf223, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf223, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf224 = buf222[1]
            assert_size_stride(buf224, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf224, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf222
            buf225 = reinterpret_tensor(buf221, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf223, buf225, 1536, 169, stream=raw_stream0)
            del buf223
            # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
            buf227 = extern_kernels.convolution(buf225, buf226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf227, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf226
            buf228 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_82, lif_forward_state_default_35], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf227, arg17_1, arg18_1, arg19_1, arg20_1, buf228, 1024, 169, stream=raw_stream0)
            del buf227
            # Topologically Sorted Source Nodes: [input_82, lif_forward_state_default_35], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf229 = torch.ops.snn_custom.lif_forward_state.default(buf228, buf182, 1.0, 0.0, 2.0, False)
            del buf182
            buf230 = buf229[0]
            assert_size_stride(buf230, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf230, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf231 = buf229[1]
            assert_size_stride(buf231, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf231, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf229
            buf232 = reinterpret_tensor(buf228, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf230, buf232, 1024, 169, stream=raw_stream0)
            del buf230
            # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
            buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf234, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf235 = reinterpret_tensor(buf232, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [input_84, lif_forward_state_default_36], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf234, arg22_1, arg23_1, arg24_1, arg25_1, buf235, 1024, 169, stream=raw_stream0)
            del buf234
            # Topologically Sorted Source Nodes: [input_84, lif_forward_state_default_36], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf236 = torch.ops.snn_custom.lif_forward_state.default(buf235, buf189, 1.0, 0.0, 2.0, False)
            del buf189
            del buf235
            buf237 = buf236[0]
            assert_size_stride(buf237, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf237, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf238 = buf236[1]
            assert_size_stride(buf238, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf238, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf236
            buf240 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_85, x_8], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf237, buf240, 36864, stream=raw_stream0)
            buf241 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_8, x_9, input_87], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf241)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_37], Original ATen: [snn_custom.lif_forward_state]
            buf242 = torch.ops.snn_custom.lif_forward_state.default(buf241, buf195, 1.0, 0.0, 2.0, False)
            del buf195
            buf243 = buf242[0]
            assert_size_stride(buf243, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf243, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf244 = buf242[1]
            assert_size_stride(buf244, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf244, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf242
            buf245 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf243, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf245)
            del buf243
            # Topologically Sorted Source Nodes: [lif_forward_state_default_38], Original ATen: [snn_custom.lif_forward_state]
            buf246 = torch.ops.snn_custom.lif_forward_state.default(buf245, buf199, 1.0, 0.0, 2.0, False)
            del buf199
            del buf245
            buf247 = buf246[0]
            assert_size_stride(buf247, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf247, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf248 = buf246[1]
            assert_size_stride(buf248, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf248, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf246
            buf249 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf247, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf249)
            del buf247
            # Topologically Sorted Source Nodes: [lif_forward_state_default_39], Original ATen: [snn_custom.lif_forward_state]
            buf250 = torch.ops.snn_custom.lif_forward_state.default(buf249, buf203, 1.0, 0.0, 2.0, False)
            del buf203
            buf251 = buf250[0]
            assert_size_stride(buf251, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf251, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf252 = buf250[1]
            assert_size_stride(buf252, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf252, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf250
            buf253 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_85, input_91], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_30.run(arg0_1, buf253, 12, 50176, stream=raw_stream0)
            buf254 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf303 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_85, input_91, getitem_102, input_109], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf254, buf303, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_85, input_91], Original ATen: [aten.select, aten.convolution]
            buf255 = extern_kernels.convolution(buf253, buf254, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf255, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf253
            del buf254
            buf256 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [input_92, lif_forward_state_default_40], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf255, arg2_1, arg3_1, arg4_1, arg5_1, buf256, 256, 3025, stream=raw_stream0)
            del buf255
            # Topologically Sorted Source Nodes: [input_92, lif_forward_state_default_40], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf257 = torch.ops.snn_custom.lif_forward_state.default(buf256, buf210, 1.0, 0.0, 2.0, False)
            del buf210
            del buf256
            buf258 = buf257[0]
            assert_size_stride(buf258, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf258, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf259 = buf257[1]
            assert_size_stride(buf259, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf259, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf257
            buf260 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf258, buf260, 256, 729, stream=raw_stream0)
            del buf258
            buf261 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf310 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_94, input_112], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf261, buf310, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
            buf262 = extern_kernels.convolution(buf260, buf261, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf262, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf260
            del buf261
            buf263 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [input_95, lif_forward_state_default_41], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf262, arg7_1, arg8_1, arg9_1, arg10_1, buf263, 768, 729, stream=raw_stream0)
            del buf262
            # Topologically Sorted Source Nodes: [input_95, lif_forward_state_default_41], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf264 = torch.ops.snn_custom.lif_forward_state.default(buf263, buf217, 1.0, 0.0, 2.0, False)
            del buf217
            del buf263
            buf265 = buf264[0]
            assert_size_stride(buf265, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf265, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf266 = buf264[1]
            assert_size_stride(buf266, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf266, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf264
            buf267 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf265, buf267, 768, 169, stream=raw_stream0)
            del buf265
            buf268 = buf219; del buf219  # reuse
            buf317 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_97, input_115], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf268, buf317, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
            buf269 = extern_kernels.convolution(buf267, buf268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf269, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf267
            del buf268
            buf270 = reinterpret_tensor(buf225, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [input_98, lif_forward_state_default_42], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf269, arg12_1, arg13_1, arg14_1, arg15_1, buf270, 1536, 169, stream=raw_stream0)
            del buf269
            # Topologically Sorted Source Nodes: [input_98, lif_forward_state_default_42], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf271 = torch.ops.snn_custom.lif_forward_state.default(buf270, buf224, 1.0, 0.0, 2.0, False)
            del buf224
            buf272 = buf271[0]
            assert_size_stride(buf272, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf272, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf273 = buf271[1]
            assert_size_stride(buf273, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf273, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf271
            buf274 = reinterpret_tensor(buf270, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf272, buf274, 1536, 169, stream=raw_stream0)
            del buf272
            buf275 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf324 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_99, input_117], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf275, buf324, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
            buf276 = extern_kernels.convolution(buf274, buf275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf276, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf274
            del buf275
            buf277 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [input_100, lif_forward_state_default_43], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf276, arg17_1, arg18_1, arg19_1, arg20_1, buf277, 1024, 169, stream=raw_stream0)
            del buf276
            # Topologically Sorted Source Nodes: [input_100, lif_forward_state_default_43], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf278 = torch.ops.snn_custom.lif_forward_state.default(buf277, buf231, 1.0, 0.0, 2.0, False)
            del buf231
            buf279 = buf278[0]
            assert_size_stride(buf279, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf279, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf280 = buf278[1]
            assert_size_stride(buf280, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf280, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf278
            buf281 = reinterpret_tensor(buf277, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf279, buf281, 1024, 169, stream=raw_stream0)
            del buf279
            buf282 = buf233; del buf233  # reuse
            buf331 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_101, input_119], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf282, buf331, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
            buf283 = extern_kernels.convolution(buf281, buf282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf283, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf282
            buf284 = reinterpret_tensor(buf281, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [input_102, lif_forward_state_default_44], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf283, arg22_1, arg23_1, arg24_1, arg25_1, buf284, 1024, 169, stream=raw_stream0)
            del buf283
            # Topologically Sorted Source Nodes: [input_102, lif_forward_state_default_44], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf285 = torch.ops.snn_custom.lif_forward_state.default(buf284, buf238, 1.0, 0.0, 2.0, False)
            del buf238
            del buf284
            buf286 = buf285[0]
            assert_size_stride(buf286, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf286, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf287 = buf285[1]
            assert_size_stride(buf287, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf287, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf285
            buf289 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [input_103, x_10], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf286, buf289, 36864, stream=raw_stream0)
            del buf286
            buf290 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_10, x_11, input_105], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf289, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf290)
            del buf289
            # Topologically Sorted Source Nodes: [lif_forward_state_default_45], Original ATen: [snn_custom.lif_forward_state]
            buf291 = torch.ops.snn_custom.lif_forward_state.default(buf290, buf244, 1.0, 0.0, 2.0, False)
            del buf244
            buf292 = buf291[0]
            assert_size_stride(buf292, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf292, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf293 = buf291[1]
            assert_size_stride(buf293, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf293, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf291
            buf294 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf292, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf294)
            del buf292
            # Topologically Sorted Source Nodes: [lif_forward_state_default_46], Original ATen: [snn_custom.lif_forward_state]
            buf295 = torch.ops.snn_custom.lif_forward_state.default(buf294, buf248, 1.0, 0.0, 2.0, False)
            del buf248
            del buf294
            buf296 = buf295[0]
            assert_size_stride(buf296, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf296, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf297 = buf295[1]
            assert_size_stride(buf297, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf297, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf295
            buf298 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf296, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf298)
            del buf296
            # Topologically Sorted Source Nodes: [lif_forward_state_default_47], Original ATen: [snn_custom.lif_forward_state]
            buf299 = torch.ops.snn_custom.lif_forward_state.default(buf298, buf252, 1.0, 0.0, 2.0, False)
            del buf252
            buf300 = buf299[0]
            assert_size_stride(buf300, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf300, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf301 = buf299[1]
            assert_size_stride(buf301, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf301, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf299
            buf302 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_102, input_109], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_31.run(arg0_1, buf302, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_102, input_109], Original ATen: [aten.select, aten.convolution]
            buf304 = extern_kernels.convolution(buf302, buf303, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf304, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf302
            del buf303
            buf305 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_110, lif_forward_state_default_48], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf304, arg2_1, arg3_1, arg4_1, arg5_1, buf305, 256, 3025, stream=raw_stream0)
            del buf304
            # Topologically Sorted Source Nodes: [input_110, lif_forward_state_default_48], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf306 = torch.ops.snn_custom.lif_forward_state.default(buf305, buf259, 1.0, 0.0, 2.0, False)
            del buf259
            del buf305
            buf307 = buf306[0]
            assert_size_stride(buf307, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf307, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf308 = buf306[1]
            assert_size_stride(buf308, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf308, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf306
            buf309 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf307, buf309, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
            buf311 = extern_kernels.convolution(buf309, buf310, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf311, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf309
            del buf310
            buf312 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_113, lif_forward_state_default_49], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf311, arg7_1, arg8_1, arg9_1, arg10_1, buf312, 768, 729, stream=raw_stream0)
            del buf311
            # Topologically Sorted Source Nodes: [input_113, lif_forward_state_default_49], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf313 = torch.ops.snn_custom.lif_forward_state.default(buf312, buf266, 1.0, 0.0, 2.0, False)
            del buf266
            del buf312
            buf314 = buf313[0]
            assert_size_stride(buf314, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf314, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf315 = buf313[1]
            assert_size_stride(buf315, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf315, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf313
            buf316 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf314, buf316, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
            buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf318, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf319 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_116, lif_forward_state_default_50], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf318, arg12_1, arg13_1, arg14_1, arg15_1, buf319, 1536, 169, stream=raw_stream0)
            del buf318
            # Topologically Sorted Source Nodes: [input_116, lif_forward_state_default_50], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf320 = torch.ops.snn_custom.lif_forward_state.default(buf319, buf273, 1.0, 0.0, 2.0, False)
            del buf273
            buf321 = buf320[0]
            assert_size_stride(buf321, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf321, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf322 = buf320[1]
            assert_size_stride(buf322, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf322, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf320
            buf323 = reinterpret_tensor(buf319, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf321, buf323, 1536, 169, stream=raw_stream0)
            del buf321
            # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
            buf325 = extern_kernels.convolution(buf323, buf324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf325, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf324
            buf326 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_118, lif_forward_state_default_51], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf325, arg17_1, arg18_1, arg19_1, arg20_1, buf326, 1024, 169, stream=raw_stream0)
            del buf325
            # Topologically Sorted Source Nodes: [input_118, lif_forward_state_default_51], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf327 = torch.ops.snn_custom.lif_forward_state.default(buf326, buf280, 1.0, 0.0, 2.0, False)
            del buf280
            buf328 = buf327[0]
            assert_size_stride(buf328, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf328, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf329 = buf327[1]
            assert_size_stride(buf329, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf329, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf327
            buf330 = reinterpret_tensor(buf326, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf328, buf330, 1024, 169, stream=raw_stream0)
            del buf328
            # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
            buf332 = extern_kernels.convolution(buf330, buf331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf332, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf333 = reinterpret_tensor(buf330, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [input_120, lif_forward_state_default_52], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf332, arg22_1, arg23_1, arg24_1, arg25_1, buf333, 1024, 169, stream=raw_stream0)
            del buf332
            # Topologically Sorted Source Nodes: [input_120, lif_forward_state_default_52], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf334 = torch.ops.snn_custom.lif_forward_state.default(buf333, buf287, 1.0, 0.0, 2.0, False)
            del buf287
            del buf333
            buf335 = buf334[0]
            assert_size_stride(buf335, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf335, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf336 = buf334[1]
            assert_size_stride(buf336, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf336, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf334
            buf338 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_121, x_12], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf335, buf338, 36864, stream=raw_stream0)
            buf339 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, input_123], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf338, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf339)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_53], Original ATen: [snn_custom.lif_forward_state]
            buf340 = torch.ops.snn_custom.lif_forward_state.default(buf339, buf293, 1.0, 0.0, 2.0, False)
            del buf293
            buf341 = buf340[0]
            assert_size_stride(buf341, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf341, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf342 = buf340[1]
            assert_size_stride(buf342, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf342, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf340
            buf343 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf341, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf343)
            del buf341
            # Topologically Sorted Source Nodes: [lif_forward_state_default_54], Original ATen: [snn_custom.lif_forward_state]
            buf344 = torch.ops.snn_custom.lif_forward_state.default(buf343, buf297, 1.0, 0.0, 2.0, False)
            del buf297
            del buf343
            buf345 = buf344[0]
            assert_size_stride(buf345, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf345, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf346 = buf344[1]
            assert_size_stride(buf346, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf346, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf344
            buf347 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf345, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf347)
            del buf345
            # Topologically Sorted Source Nodes: [lif_forward_state_default_55], Original ATen: [snn_custom.lif_forward_state]
            buf348 = torch.ops.snn_custom.lif_forward_state.default(buf347, buf301, 1.0, 0.0, 2.0, False)
            del buf301
            buf349 = buf348[0]
            assert_size_stride(buf349, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf349, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf350 = buf348[1]
            assert_size_stride(buf350, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf350, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf348
            buf351 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_119, input_127], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_32.run(arg0_1, buf351, 12, 50176, stream=raw_stream0)
            buf352 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf401 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_119, input_127, getitem_136, input_145], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf352, buf401, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_119, input_127], Original ATen: [aten.select, aten.convolution]
            buf353 = extern_kernels.convolution(buf351, buf352, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf353, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf351
            del buf352
            buf354 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [input_128, lif_forward_state_default_56], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf353, arg2_1, arg3_1, arg4_1, arg5_1, buf354, 256, 3025, stream=raw_stream0)
            del buf353
            # Topologically Sorted Source Nodes: [input_128, lif_forward_state_default_56], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf355 = torch.ops.snn_custom.lif_forward_state.default(buf354, buf308, 1.0, 0.0, 2.0, False)
            del buf308
            del buf354
            buf356 = buf355[0]
            assert_size_stride(buf356, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf356, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf357 = buf355[1]
            assert_size_stride(buf357, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf357, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf355
            buf358 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf356, buf358, 256, 729, stream=raw_stream0)
            del buf356
            buf359 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf408 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_130, input_148], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf359, buf408, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
            buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf360, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf358
            del buf359
            buf361 = buf314; del buf314  # reuse
            # Topologically Sorted Source Nodes: [input_131, lif_forward_state_default_57], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf360, arg7_1, arg8_1, arg9_1, arg10_1, buf361, 768, 729, stream=raw_stream0)
            del buf360
            # Topologically Sorted Source Nodes: [input_131, lif_forward_state_default_57], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf362 = torch.ops.snn_custom.lif_forward_state.default(buf361, buf315, 1.0, 0.0, 2.0, False)
            del buf315
            del buf361
            buf363 = buf362[0]
            assert_size_stride(buf363, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf363, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf364 = buf362[1]
            assert_size_stride(buf364, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf364, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf362
            buf365 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [input_132], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf363, buf365, 768, 169, stream=raw_stream0)
            del buf363
            buf366 = buf317; del buf317  # reuse
            buf415 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_133, input_151], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf366, buf415, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
            buf367 = extern_kernels.convolution(buf365, buf366, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf367, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf365
            del buf366
            buf368 = reinterpret_tensor(buf323, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [input_134, lif_forward_state_default_58], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf367, arg12_1, arg13_1, arg14_1, arg15_1, buf368, 1536, 169, stream=raw_stream0)
            del buf367
            # Topologically Sorted Source Nodes: [input_134, lif_forward_state_default_58], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf369 = torch.ops.snn_custom.lif_forward_state.default(buf368, buf322, 1.0, 0.0, 2.0, False)
            del buf322
            buf370 = buf369[0]
            assert_size_stride(buf370, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf370, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf371 = buf369[1]
            assert_size_stride(buf371, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf371, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf369
            buf372 = reinterpret_tensor(buf368, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf370, buf372, 1536, 169, stream=raw_stream0)
            del buf370
            buf373 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf422 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_135, input_153], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf373, buf422, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
            buf374 = extern_kernels.convolution(buf372, buf373, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf374, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf372
            del buf373
            buf375 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [input_136, lif_forward_state_default_59], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf374, arg17_1, arg18_1, arg19_1, arg20_1, buf375, 1024, 169, stream=raw_stream0)
            del buf374
            # Topologically Sorted Source Nodes: [input_136, lif_forward_state_default_59], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf376 = torch.ops.snn_custom.lif_forward_state.default(buf375, buf329, 1.0, 0.0, 2.0, False)
            del buf329
            buf377 = buf376[0]
            assert_size_stride(buf377, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf377, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf378 = buf376[1]
            assert_size_stride(buf378, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf378, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf376
            buf379 = reinterpret_tensor(buf375, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf375  # reuse
            # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf377, buf379, 1024, 169, stream=raw_stream0)
            del buf377
            buf380 = buf331; del buf331  # reuse
            buf429 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_137, input_155], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf380, buf429, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
            buf381 = extern_kernels.convolution(buf379, buf380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf381, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf380
            buf382 = reinterpret_tensor(buf379, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [input_138, lif_forward_state_default_60], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf381, arg22_1, arg23_1, arg24_1, arg25_1, buf382, 1024, 169, stream=raw_stream0)
            del buf381
            # Topologically Sorted Source Nodes: [input_138, lif_forward_state_default_60], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf383 = torch.ops.snn_custom.lif_forward_state.default(buf382, buf336, 1.0, 0.0, 2.0, False)
            del buf336
            del buf382
            buf384 = buf383[0]
            assert_size_stride(buf384, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf384, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf385 = buf383[1]
            assert_size_stride(buf385, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf385, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf383
            buf387 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [input_139, x_14], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf384, buf387, 36864, stream=raw_stream0)
            del buf384
            buf388 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, input_141], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf387, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf388)
            del buf387
            # Topologically Sorted Source Nodes: [lif_forward_state_default_61], Original ATen: [snn_custom.lif_forward_state]
            buf389 = torch.ops.snn_custom.lif_forward_state.default(buf388, buf342, 1.0, 0.0, 2.0, False)
            del buf342
            buf390 = buf389[0]
            assert_size_stride(buf390, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf390, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf391 = buf389[1]
            assert_size_stride(buf391, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf391, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf389
            buf392 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf390, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf392)
            del buf390
            # Topologically Sorted Source Nodes: [lif_forward_state_default_62], Original ATen: [snn_custom.lif_forward_state]
            buf393 = torch.ops.snn_custom.lif_forward_state.default(buf392, buf346, 1.0, 0.0, 2.0, False)
            del buf346
            del buf392
            buf394 = buf393[0]
            assert_size_stride(buf394, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf394, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf395 = buf393[1]
            assert_size_stride(buf395, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf395, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf393
            buf396 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf394, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf396)
            del buf394
            # Topologically Sorted Source Nodes: [lif_forward_state_default_63], Original ATen: [snn_custom.lif_forward_state]
            buf397 = torch.ops.snn_custom.lif_forward_state.default(buf396, buf350, 1.0, 0.0, 2.0, False)
            del buf350
            buf398 = buf397[0]
            assert_size_stride(buf398, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf398, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf399 = buf397[1]
            assert_size_stride(buf399, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf399, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf397
            buf400 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_136, input_145], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_33.run(arg0_1, buf400, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_136, input_145], Original ATen: [aten.select, aten.convolution]
            buf402 = extern_kernels.convolution(buf400, buf401, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf402, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf400
            del buf401
            buf403 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_146, lif_forward_state_default_64], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf402, arg2_1, arg3_1, arg4_1, arg5_1, buf403, 256, 3025, stream=raw_stream0)
            del buf402
            # Topologically Sorted Source Nodes: [input_146, lif_forward_state_default_64], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf404 = torch.ops.snn_custom.lif_forward_state.default(buf403, buf357, 1.0, 0.0, 2.0, False)
            del buf357
            del buf403
            buf405 = buf404[0]
            assert_size_stride(buf405, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf405, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf406 = buf404[1]
            assert_size_stride(buf406, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf406, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf404
            buf407 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf405, buf407, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
            buf409 = extern_kernels.convolution(buf407, buf408, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf409, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf407
            del buf408
            buf410 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_149, lif_forward_state_default_65], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf409, arg7_1, arg8_1, arg9_1, arg10_1, buf410, 768, 729, stream=raw_stream0)
            del buf409
            # Topologically Sorted Source Nodes: [input_149, lif_forward_state_default_65], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf411 = torch.ops.snn_custom.lif_forward_state.default(buf410, buf364, 1.0, 0.0, 2.0, False)
            del buf364
            del buf410
            buf412 = buf411[0]
            assert_size_stride(buf412, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf412, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf413 = buf411[1]
            assert_size_stride(buf413, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf413, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf411
            buf414 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf412, buf414, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
            buf416 = extern_kernels.convolution(buf414, buf415, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf416, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf417 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_152, lif_forward_state_default_66], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf416, arg12_1, arg13_1, arg14_1, arg15_1, buf417, 1536, 169, stream=raw_stream0)
            del buf416
            # Topologically Sorted Source Nodes: [input_152, lif_forward_state_default_66], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf418 = torch.ops.snn_custom.lif_forward_state.default(buf417, buf371, 1.0, 0.0, 2.0, False)
            del buf371
            buf419 = buf418[0]
            assert_size_stride(buf419, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf419, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf420 = buf418[1]
            assert_size_stride(buf420, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf420, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf418
            buf421 = reinterpret_tensor(buf417, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf417  # reuse
            # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf419, buf421, 1536, 169, stream=raw_stream0)
            del buf419
            # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
            buf423 = extern_kernels.convolution(buf421, buf422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf423, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf422
            buf424 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_154, lif_forward_state_default_67], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf423, arg17_1, arg18_1, arg19_1, arg20_1, buf424, 1024, 169, stream=raw_stream0)
            del buf423
            # Topologically Sorted Source Nodes: [input_154, lif_forward_state_default_67], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf425 = torch.ops.snn_custom.lif_forward_state.default(buf424, buf378, 1.0, 0.0, 2.0, False)
            del buf378
            buf426 = buf425[0]
            assert_size_stride(buf426, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf426, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf427 = buf425[1]
            assert_size_stride(buf427, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf427, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf425
            buf428 = reinterpret_tensor(buf424, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf426, buf428, 1024, 169, stream=raw_stream0)
            del buf426
            # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.convolution]
            buf430 = extern_kernels.convolution(buf428, buf429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf430, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf431 = reinterpret_tensor(buf428, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf428  # reuse
            # Topologically Sorted Source Nodes: [input_156, lif_forward_state_default_68], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf430, arg22_1, arg23_1, arg24_1, arg25_1, buf431, 1024, 169, stream=raw_stream0)
            del buf430
            # Topologically Sorted Source Nodes: [input_156, lif_forward_state_default_68], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf432 = torch.ops.snn_custom.lif_forward_state.default(buf431, buf385, 1.0, 0.0, 2.0, False)
            del buf385
            del buf431
            buf433 = buf432[0]
            assert_size_stride(buf433, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf433, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf434 = buf432[1]
            assert_size_stride(buf434, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf434, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf432
            buf436 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_157, x_16], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf433, buf436, 36864, stream=raw_stream0)
            buf437 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17, input_159], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf436, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf437)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_69], Original ATen: [snn_custom.lif_forward_state]
            buf438 = torch.ops.snn_custom.lif_forward_state.default(buf437, buf391, 1.0, 0.0, 2.0, False)
            del buf391
            buf439 = buf438[0]
            assert_size_stride(buf439, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf439, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf440 = buf438[1]
            assert_size_stride(buf440, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf440, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf438
            buf441 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf439, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf441)
            del buf439
            # Topologically Sorted Source Nodes: [lif_forward_state_default_70], Original ATen: [snn_custom.lif_forward_state]
            buf442 = torch.ops.snn_custom.lif_forward_state.default(buf441, buf395, 1.0, 0.0, 2.0, False)
            del buf395
            del buf441
            buf443 = buf442[0]
            assert_size_stride(buf443, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf443, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf444 = buf442[1]
            assert_size_stride(buf444, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf444, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf442
            buf445 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf443, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf445)
            del buf443
            # Topologically Sorted Source Nodes: [lif_forward_state_default_71], Original ATen: [snn_custom.lif_forward_state]
            buf446 = torch.ops.snn_custom.lif_forward_state.default(buf445, buf399, 1.0, 0.0, 2.0, False)
            del buf399
            buf447 = buf446[0]
            assert_size_stride(buf447, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf447, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf448 = buf446[1]
            assert_size_stride(buf448, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf448, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf446
            buf449 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_153, input_163], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_34.run(arg0_1, buf449, 12, 50176, stream=raw_stream0)
            buf450 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf499 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_153, input_163, getitem_170, input_181], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf450, buf499, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_153, input_163], Original ATen: [aten.select, aten.convolution]
            buf451 = extern_kernels.convolution(buf449, buf450, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf451, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf449
            del buf450
            buf452 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [input_164, lif_forward_state_default_72], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf451, arg2_1, arg3_1, arg4_1, arg5_1, buf452, 256, 3025, stream=raw_stream0)
            del buf451
            # Topologically Sorted Source Nodes: [input_164, lif_forward_state_default_72], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf453 = torch.ops.snn_custom.lif_forward_state.default(buf452, buf406, 1.0, 0.0, 2.0, False)
            del buf406
            del buf452
            buf454 = buf453[0]
            assert_size_stride(buf454, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf454, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf455 = buf453[1]
            assert_size_stride(buf455, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf455, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf453
            buf456 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf454, buf456, 256, 729, stream=raw_stream0)
            del buf454
            buf457 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf506 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_166, input_184], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf457, buf506, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
            buf458 = extern_kernels.convolution(buf456, buf457, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf458, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf456
            del buf457
            buf459 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [input_167, lif_forward_state_default_73], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf458, arg7_1, arg8_1, arg9_1, arg10_1, buf459, 768, 729, stream=raw_stream0)
            del buf458
            # Topologically Sorted Source Nodes: [input_167, lif_forward_state_default_73], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf460 = torch.ops.snn_custom.lif_forward_state.default(buf459, buf413, 1.0, 0.0, 2.0, False)
            del buf413
            del buf459
            buf461 = buf460[0]
            assert_size_stride(buf461, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf461, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf462 = buf460[1]
            assert_size_stride(buf462, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf462, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf460
            buf463 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf461, buf463, 768, 169, stream=raw_stream0)
            del buf461
            buf464 = buf415; del buf415  # reuse
            buf513 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_169, input_187], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf464, buf513, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
            buf465 = extern_kernels.convolution(buf463, buf464, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf465, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf463
            del buf464
            buf466 = reinterpret_tensor(buf421, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf421  # reuse
            # Topologically Sorted Source Nodes: [input_170, lif_forward_state_default_74], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf465, arg12_1, arg13_1, arg14_1, arg15_1, buf466, 1536, 169, stream=raw_stream0)
            del buf465
            # Topologically Sorted Source Nodes: [input_170, lif_forward_state_default_74], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf467 = torch.ops.snn_custom.lif_forward_state.default(buf466, buf420, 1.0, 0.0, 2.0, False)
            del buf420
            buf468 = buf467[0]
            assert_size_stride(buf468, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf468, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf469 = buf467[1]
            assert_size_stride(buf469, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf469, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf467
            buf470 = reinterpret_tensor(buf466, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf468, buf470, 1536, 169, stream=raw_stream0)
            del buf468
            buf471 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf520 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_171, input_189], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf471, buf520, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
            buf472 = extern_kernels.convolution(buf470, buf471, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf472, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf470
            del buf471
            buf473 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [input_172, lif_forward_state_default_75], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf472, arg17_1, arg18_1, arg19_1, arg20_1, buf473, 1024, 169, stream=raw_stream0)
            del buf472
            # Topologically Sorted Source Nodes: [input_172, lif_forward_state_default_75], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf474 = torch.ops.snn_custom.lif_forward_state.default(buf473, buf427, 1.0, 0.0, 2.0, False)
            del buf427
            buf475 = buf474[0]
            assert_size_stride(buf475, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf475, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf476 = buf474[1]
            assert_size_stride(buf476, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf476, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf474
            buf477 = reinterpret_tensor(buf473, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf475, buf477, 1024, 169, stream=raw_stream0)
            del buf475
            buf478 = buf429; del buf429  # reuse
            buf527 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_173, input_191], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf478, buf527, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.convolution]
            buf479 = extern_kernels.convolution(buf477, buf478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf479, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf478
            buf480 = reinterpret_tensor(buf477, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [input_174, lif_forward_state_default_76], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf479, arg22_1, arg23_1, arg24_1, arg25_1, buf480, 1024, 169, stream=raw_stream0)
            del buf479
            # Topologically Sorted Source Nodes: [input_174, lif_forward_state_default_76], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf481 = torch.ops.snn_custom.lif_forward_state.default(buf480, buf434, 1.0, 0.0, 2.0, False)
            del buf434
            del buf480
            buf482 = buf481[0]
            assert_size_stride(buf482, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf482, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf483 = buf481[1]
            assert_size_stride(buf483, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf483, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf481
            buf485 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [input_175, x_18], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf482, buf485, 36864, stream=raw_stream0)
            del buf482
            buf486 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_18, x_19, input_177], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf485, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf486)
            del buf485
            # Topologically Sorted Source Nodes: [lif_forward_state_default_77], Original ATen: [snn_custom.lif_forward_state]
            buf487 = torch.ops.snn_custom.lif_forward_state.default(buf486, buf440, 1.0, 0.0, 2.0, False)
            del buf440
            buf488 = buf487[0]
            assert_size_stride(buf488, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf488, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf489 = buf487[1]
            assert_size_stride(buf489, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf489, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf487
            buf490 = buf486; del buf486  # reuse
            # Topologically Sorted Source Nodes: [input_179], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf488, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf490)
            del buf488
            # Topologically Sorted Source Nodes: [lif_forward_state_default_78], Original ATen: [snn_custom.lif_forward_state]
            buf491 = torch.ops.snn_custom.lif_forward_state.default(buf490, buf444, 1.0, 0.0, 2.0, False)
            del buf444
            del buf490
            buf492 = buf491[0]
            assert_size_stride(buf492, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf492, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf493 = buf491[1]
            assert_size_stride(buf493, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf493, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf491
            buf494 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf492, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf494)
            del buf492
            # Topologically Sorted Source Nodes: [lif_forward_state_default_79], Original ATen: [snn_custom.lif_forward_state]
            buf495 = torch.ops.snn_custom.lif_forward_state.default(buf494, buf448, 1.0, 0.0, 2.0, False)
            del buf448
            buf496 = buf495[0]
            assert_size_stride(buf496, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf496, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf497 = buf495[1]
            assert_size_stride(buf497, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf497, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf495
            buf498 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_170, input_181], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_35.run(arg0_1, buf498, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_170, input_181], Original ATen: [aten.select, aten.convolution]
            buf500 = extern_kernels.convolution(buf498, buf499, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf500, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf498
            del buf499
            buf501 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_182, lif_forward_state_default_80], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf500, arg2_1, arg3_1, arg4_1, arg5_1, buf501, 256, 3025, stream=raw_stream0)
            del buf500
            # Topologically Sorted Source Nodes: [input_182, lif_forward_state_default_80], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf502 = torch.ops.snn_custom.lif_forward_state.default(buf501, buf455, 1.0, 0.0, 2.0, False)
            del buf455
            del buf501
            buf503 = buf502[0]
            assert_size_stride(buf503, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf503, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf504 = buf502[1]
            assert_size_stride(buf504, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf504, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf502
            buf505 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf503, buf505, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
            buf507 = extern_kernels.convolution(buf505, buf506, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf507, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf505
            del buf506
            buf508 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_185, lif_forward_state_default_81], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf507, arg7_1, arg8_1, arg9_1, arg10_1, buf508, 768, 729, stream=raw_stream0)
            del buf507
            # Topologically Sorted Source Nodes: [input_185, lif_forward_state_default_81], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf509 = torch.ops.snn_custom.lif_forward_state.default(buf508, buf462, 1.0, 0.0, 2.0, False)
            del buf462
            del buf508
            buf510 = buf509[0]
            assert_size_stride(buf510, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf510, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf511 = buf509[1]
            assert_size_stride(buf511, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf511, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf509
            buf512 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf510, buf512, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_187], Original ATen: [aten.convolution]
            buf514 = extern_kernels.convolution(buf512, buf513, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf514, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf515 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_188, lif_forward_state_default_82], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf514, arg12_1, arg13_1, arg14_1, arg15_1, buf515, 1536, 169, stream=raw_stream0)
            del buf514
            # Topologically Sorted Source Nodes: [input_188, lif_forward_state_default_82], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf516 = torch.ops.snn_custom.lif_forward_state.default(buf515, buf469, 1.0, 0.0, 2.0, False)
            del buf469
            buf517 = buf516[0]
            assert_size_stride(buf517, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf517, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf518 = buf516[1]
            assert_size_stride(buf518, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf518, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf516
            buf519 = reinterpret_tensor(buf515, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf515  # reuse
            # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf517, buf519, 1536, 169, stream=raw_stream0)
            del buf517
            # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
            buf521 = extern_kernels.convolution(buf519, buf520, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf521, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf520
            buf522 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_190, lif_forward_state_default_83], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf521, arg17_1, arg18_1, arg19_1, arg20_1, buf522, 1024, 169, stream=raw_stream0)
            del buf521
            # Topologically Sorted Source Nodes: [input_190, lif_forward_state_default_83], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf523 = torch.ops.snn_custom.lif_forward_state.default(buf522, buf476, 1.0, 0.0, 2.0, False)
            del buf476
            buf524 = buf523[0]
            assert_size_stride(buf524, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf524, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf525 = buf523[1]
            assert_size_stride(buf525, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf525, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf523
            buf526 = reinterpret_tensor(buf522, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf522  # reuse
            # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf524, buf526, 1024, 169, stream=raw_stream0)
            del buf524
            # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
            buf528 = extern_kernels.convolution(buf526, buf527, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf528, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf529 = reinterpret_tensor(buf526, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf526  # reuse
            # Topologically Sorted Source Nodes: [input_192, lif_forward_state_default_84], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf528, arg22_1, arg23_1, arg24_1, arg25_1, buf529, 1024, 169, stream=raw_stream0)
            del buf528
            # Topologically Sorted Source Nodes: [input_192, lif_forward_state_default_84], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf530 = torch.ops.snn_custom.lif_forward_state.default(buf529, buf483, 1.0, 0.0, 2.0, False)
            del buf483
            del buf529
            buf531 = buf530[0]
            assert_size_stride(buf531, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf531, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf532 = buf530[1]
            assert_size_stride(buf532, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf532, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf530
            buf534 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_193, x_20], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf531, buf534, 36864, stream=raw_stream0)
            buf535 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, input_195], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf534, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf535)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_85], Original ATen: [snn_custom.lif_forward_state]
            buf536 = torch.ops.snn_custom.lif_forward_state.default(buf535, buf489, 1.0, 0.0, 2.0, False)
            del buf489
            buf537 = buf536[0]
            assert_size_stride(buf537, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf537, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf538 = buf536[1]
            assert_size_stride(buf538, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf538, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf536
            buf539 = buf535; del buf535  # reuse
            # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf537, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf539)
            del buf537
            # Topologically Sorted Source Nodes: [lif_forward_state_default_86], Original ATen: [snn_custom.lif_forward_state]
            buf540 = torch.ops.snn_custom.lif_forward_state.default(buf539, buf493, 1.0, 0.0, 2.0, False)
            del buf493
            del buf539
            buf541 = buf540[0]
            assert_size_stride(buf541, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf541, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf542 = buf540[1]
            assert_size_stride(buf542, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf542, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf540
            buf543 = buf494; del buf494  # reuse
            # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf541, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf543)
            del buf541
            # Topologically Sorted Source Nodes: [lif_forward_state_default_87], Original ATen: [snn_custom.lif_forward_state]
            buf544 = torch.ops.snn_custom.lif_forward_state.default(buf543, buf497, 1.0, 0.0, 2.0, False)
            del buf497
            buf545 = buf544[0]
            assert_size_stride(buf545, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf545, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf546 = buf544[1]
            assert_size_stride(buf546, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf546, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf544
            buf547 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_187, input_199], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_36.run(arg0_1, buf547, 12, 50176, stream=raw_stream0)
            buf548 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf597 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_187, input_199, getitem_204, input_217], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf548, buf597, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_187, input_199], Original ATen: [aten.select, aten.convolution]
            buf549 = extern_kernels.convolution(buf547, buf548, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf549, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf547
            del buf548
            buf550 = buf503; del buf503  # reuse
            # Topologically Sorted Source Nodes: [input_200, lif_forward_state_default_88], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf549, arg2_1, arg3_1, arg4_1, arg5_1, buf550, 256, 3025, stream=raw_stream0)
            del buf549
            # Topologically Sorted Source Nodes: [input_200, lif_forward_state_default_88], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf551 = torch.ops.snn_custom.lif_forward_state.default(buf550, buf504, 1.0, 0.0, 2.0, False)
            del buf504
            del buf550
            buf552 = buf551[0]
            assert_size_stride(buf552, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf552, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf553 = buf551[1]
            assert_size_stride(buf553, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf553, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf551
            buf554 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf552, buf554, 256, 729, stream=raw_stream0)
            del buf552
            buf555 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf604 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_202, input_220], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf555, buf604, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten.convolution]
            buf556 = extern_kernels.convolution(buf554, buf555, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf556, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf554
            del buf555
            buf557 = buf510; del buf510  # reuse
            # Topologically Sorted Source Nodes: [input_203, lif_forward_state_default_89], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf556, arg7_1, arg8_1, arg9_1, arg10_1, buf557, 768, 729, stream=raw_stream0)
            del buf556
            # Topologically Sorted Source Nodes: [input_203, lif_forward_state_default_89], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf558 = torch.ops.snn_custom.lif_forward_state.default(buf557, buf511, 1.0, 0.0, 2.0, False)
            del buf511
            del buf557
            buf559 = buf558[0]
            assert_size_stride(buf559, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf559, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf560 = buf558[1]
            assert_size_stride(buf560, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf560, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf558
            buf561 = buf512; del buf512  # reuse
            # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf559, buf561, 768, 169, stream=raw_stream0)
            del buf559
            buf562 = buf513; del buf513  # reuse
            buf611 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_205, input_223], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf562, buf611, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.convolution]
            buf563 = extern_kernels.convolution(buf561, buf562, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf563, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf561
            del buf562
            buf564 = reinterpret_tensor(buf519, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [input_206, lif_forward_state_default_90], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf563, arg12_1, arg13_1, arg14_1, arg15_1, buf564, 1536, 169, stream=raw_stream0)
            del buf563
            # Topologically Sorted Source Nodes: [input_206, lif_forward_state_default_90], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf565 = torch.ops.snn_custom.lif_forward_state.default(buf564, buf518, 1.0, 0.0, 2.0, False)
            del buf518
            buf566 = buf565[0]
            assert_size_stride(buf566, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf566, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf567 = buf565[1]
            assert_size_stride(buf567, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf567, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf565
            buf568 = reinterpret_tensor(buf564, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf564  # reuse
            # Topologically Sorted Source Nodes: [input_207], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf566, buf568, 1536, 169, stream=raw_stream0)
            del buf566
            buf569 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf618 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_207, input_225], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf569, buf618, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_207], Original ATen: [aten.convolution]
            buf570 = extern_kernels.convolution(buf568, buf569, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf570, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf568
            del buf569
            buf571 = buf531; del buf531  # reuse
            # Topologically Sorted Source Nodes: [input_208, lif_forward_state_default_91], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf570, arg17_1, arg18_1, arg19_1, arg20_1, buf571, 1024, 169, stream=raw_stream0)
            del buf570
            # Topologically Sorted Source Nodes: [input_208, lif_forward_state_default_91], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf572 = torch.ops.snn_custom.lif_forward_state.default(buf571, buf525, 1.0, 0.0, 2.0, False)
            del buf525
            buf573 = buf572[0]
            assert_size_stride(buf573, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf573, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf574 = buf572[1]
            assert_size_stride(buf574, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf574, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf572
            buf575 = reinterpret_tensor(buf571, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf571  # reuse
            # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf573, buf575, 1024, 169, stream=raw_stream0)
            del buf573
            buf576 = buf527; del buf527  # reuse
            buf625 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_209, input_227], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf576, buf625, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
            buf577 = extern_kernels.convolution(buf575, buf576, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf577, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf576
            buf578 = reinterpret_tensor(buf575, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf575  # reuse
            # Topologically Sorted Source Nodes: [input_210, lif_forward_state_default_92], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf577, arg22_1, arg23_1, arg24_1, arg25_1, buf578, 1024, 169, stream=raw_stream0)
            del buf577
            # Topologically Sorted Source Nodes: [input_210, lif_forward_state_default_92], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf579 = torch.ops.snn_custom.lif_forward_state.default(buf578, buf532, 1.0, 0.0, 2.0, False)
            del buf532
            del buf578
            buf580 = buf579[0]
            assert_size_stride(buf580, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf580, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf581 = buf579[1]
            assert_size_stride(buf581, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf581, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf579
            buf583 = buf534; del buf534  # reuse
            # Topologically Sorted Source Nodes: [input_211, x_22], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf580, buf583, 36864, stream=raw_stream0)
            del buf580
            buf584 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, input_213], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf583, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf584)
            del buf583
            # Topologically Sorted Source Nodes: [lif_forward_state_default_93], Original ATen: [snn_custom.lif_forward_state]
            buf585 = torch.ops.snn_custom.lif_forward_state.default(buf584, buf538, 1.0, 0.0, 2.0, False)
            del buf538
            buf586 = buf585[0]
            assert_size_stride(buf586, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf586, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf587 = buf585[1]
            assert_size_stride(buf587, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf587, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf585
            buf588 = buf584; del buf584  # reuse
            # Topologically Sorted Source Nodes: [input_215], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf586, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf588)
            del buf586
            # Topologically Sorted Source Nodes: [lif_forward_state_default_94], Original ATen: [snn_custom.lif_forward_state]
            buf589 = torch.ops.snn_custom.lif_forward_state.default(buf588, buf542, 1.0, 0.0, 2.0, False)
            del buf542
            del buf588
            buf590 = buf589[0]
            assert_size_stride(buf590, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf590, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf591 = buf589[1]
            assert_size_stride(buf591, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf591, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf589
            buf592 = buf543; del buf543  # reuse
            # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf590, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf592)
            del buf590
            # Topologically Sorted Source Nodes: [lif_forward_state_default_95], Original ATen: [snn_custom.lif_forward_state]
            buf593 = torch.ops.snn_custom.lif_forward_state.default(buf592, buf546, 1.0, 0.0, 2.0, False)
            del buf546
            buf594 = buf593[0]
            assert_size_stride(buf594, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf594, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf595 = buf593[1]
            assert_size_stride(buf595, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf595, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf593
            buf596 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_204, input_217], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_37.run(arg0_1, buf596, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_204, input_217], Original ATen: [aten.select, aten.convolution]
            buf598 = extern_kernels.convolution(buf596, buf597, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf598, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf596
            del buf597
            buf599 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_218, lif_forward_state_default_96], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf598, arg2_1, arg3_1, arg4_1, arg5_1, buf599, 256, 3025, stream=raw_stream0)
            del buf598
            # Topologically Sorted Source Nodes: [input_218, lif_forward_state_default_96], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf600 = torch.ops.snn_custom.lif_forward_state.default(buf599, buf553, 1.0, 0.0, 2.0, False)
            del buf553
            del buf599
            buf601 = buf600[0]
            assert_size_stride(buf601, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf601, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf602 = buf600[1]
            assert_size_stride(buf602, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf602, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf600
            buf603 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf601, buf603, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
            buf605 = extern_kernels.convolution(buf603, buf604, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf605, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf603
            del buf604
            buf606 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_221, lif_forward_state_default_97], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf605, arg7_1, arg8_1, arg9_1, arg10_1, buf606, 768, 729, stream=raw_stream0)
            del buf605
            # Topologically Sorted Source Nodes: [input_221, lif_forward_state_default_97], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf607 = torch.ops.snn_custom.lif_forward_state.default(buf606, buf560, 1.0, 0.0, 2.0, False)
            del buf560
            del buf606
            buf608 = buf607[0]
            assert_size_stride(buf608, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf608, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf609 = buf607[1]
            assert_size_stride(buf609, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf609, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf607
            buf610 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf608, buf610, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
            buf612 = extern_kernels.convolution(buf610, buf611, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf612, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf613 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_224, lif_forward_state_default_98], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf612, arg12_1, arg13_1, arg14_1, arg15_1, buf613, 1536, 169, stream=raw_stream0)
            del buf612
            # Topologically Sorted Source Nodes: [input_224, lif_forward_state_default_98], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf614 = torch.ops.snn_custom.lif_forward_state.default(buf613, buf567, 1.0, 0.0, 2.0, False)
            del buf567
            buf615 = buf614[0]
            assert_size_stride(buf615, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf615, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf616 = buf614[1]
            assert_size_stride(buf616, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf616, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf614
            buf617 = reinterpret_tensor(buf613, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf613  # reuse
            # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf615, buf617, 1536, 169, stream=raw_stream0)
            del buf615
            # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.convolution]
            buf619 = extern_kernels.convolution(buf617, buf618, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf619, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf618
            buf620 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_226, lif_forward_state_default_99], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf619, arg17_1, arg18_1, arg19_1, arg20_1, buf620, 1024, 169, stream=raw_stream0)
            del buf619
            # Topologically Sorted Source Nodes: [input_226, lif_forward_state_default_99], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf621 = torch.ops.snn_custom.lif_forward_state.default(buf620, buf574, 1.0, 0.0, 2.0, False)
            del buf574
            buf622 = buf621[0]
            assert_size_stride(buf622, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf622, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf623 = buf621[1]
            assert_size_stride(buf623, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf623, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf621
            buf624 = reinterpret_tensor(buf620, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf620  # reuse
            # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf622, buf624, 1024, 169, stream=raw_stream0)
            del buf622
            # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
            buf626 = extern_kernels.convolution(buf624, buf625, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf626, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf627 = reinterpret_tensor(buf624, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf624  # reuse
            # Topologically Sorted Source Nodes: [input_228, lif_forward_state_default_100], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf626, arg22_1, arg23_1, arg24_1, arg25_1, buf627, 1024, 169, stream=raw_stream0)
            del buf626
            # Topologically Sorted Source Nodes: [input_228, lif_forward_state_default_100], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf628 = torch.ops.snn_custom.lif_forward_state.default(buf627, buf581, 1.0, 0.0, 2.0, False)
            del buf581
            del buf627
            buf629 = buf628[0]
            assert_size_stride(buf629, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf629, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf630 = buf628[1]
            assert_size_stride(buf630, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf630, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf628
            buf632 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_229, x_24], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf629, buf632, 36864, stream=raw_stream0)
            buf633 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_24, x_25, input_231], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf632, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf633)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_101], Original ATen: [snn_custom.lif_forward_state]
            buf634 = torch.ops.snn_custom.lif_forward_state.default(buf633, buf587, 1.0, 0.0, 2.0, False)
            del buf587
            buf635 = buf634[0]
            assert_size_stride(buf635, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf635, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf636 = buf634[1]
            assert_size_stride(buf636, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf636, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf634
            buf637 = buf633; del buf633  # reuse
            # Topologically Sorted Source Nodes: [input_233], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf635, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf637)
            del buf635
            # Topologically Sorted Source Nodes: [lif_forward_state_default_102], Original ATen: [snn_custom.lif_forward_state]
            buf638 = torch.ops.snn_custom.lif_forward_state.default(buf637, buf591, 1.0, 0.0, 2.0, False)
            del buf591
            del buf637
            buf639 = buf638[0]
            assert_size_stride(buf639, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf639, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf640 = buf638[1]
            assert_size_stride(buf640, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf640, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf638
            buf641 = buf592; del buf592  # reuse
            # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf639, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf641)
            del buf639
            # Topologically Sorted Source Nodes: [lif_forward_state_default_103], Original ATen: [snn_custom.lif_forward_state]
            buf642 = torch.ops.snn_custom.lif_forward_state.default(buf641, buf595, 1.0, 0.0, 2.0, False)
            del buf595
            buf643 = buf642[0]
            assert_size_stride(buf643, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf643, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf644 = buf642[1]
            assert_size_stride(buf644, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf644, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf642
            buf645 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_221, input_235], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_38.run(arg0_1, buf645, 12, 50176, stream=raw_stream0)
            buf646 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            buf695 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_221, input_235, getitem_238, input_253], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_22.run(arg1_1, buf646, buf695, 192, 121, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_221, input_235], Original ATen: [aten.select, aten.convolution]
            buf647 = extern_kernels.convolution(buf645, buf646, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf647, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf645
            del buf646
            buf648 = buf601; del buf601  # reuse
            # Topologically Sorted Source Nodes: [input_236, lif_forward_state_default_104], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf647, arg2_1, arg3_1, arg4_1, arg5_1, buf648, 256, 3025, stream=raw_stream0)
            del buf647
            # Topologically Sorted Source Nodes: [input_236, lif_forward_state_default_104], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf649 = torch.ops.snn_custom.lif_forward_state.default(buf648, buf602, 1.0, 0.0, 2.0, False)
            del buf602
            del buf648
            buf650 = buf649[0]
            assert_size_stride(buf650, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf650, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf651 = buf649[1]
            assert_size_stride(buf651, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf651, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf649
            buf652 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_237], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf650, buf652, 256, 729, stream=raw_stream0)
            del buf650
            buf653 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            buf702 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_238, input_256], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_23.run(arg6_1, buf653, buf702, 12288, 25, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
            buf654 = extern_kernels.convolution(buf652, buf653, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf654, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf652
            del buf653
            buf655 = buf608; del buf608  # reuse
            # Topologically Sorted Source Nodes: [input_239, lif_forward_state_default_105], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf654, arg7_1, arg8_1, arg9_1, arg10_1, buf655, 768, 729, stream=raw_stream0)
            del buf654
            # Topologically Sorted Source Nodes: [input_239, lif_forward_state_default_105], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf656 = torch.ops.snn_custom.lif_forward_state.default(buf655, buf609, 1.0, 0.0, 2.0, False)
            del buf609
            del buf655
            buf657 = buf656[0]
            assert_size_stride(buf657, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf657, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf658 = buf656[1]
            assert_size_stride(buf658, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf658, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf656
            buf659 = buf610; del buf610  # reuse
            # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf657, buf659, 768, 169, stream=raw_stream0)
            del buf657
            buf660 = buf611; del buf611  # reuse
            buf709 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_241, input_259], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_24.run(arg11_1, buf660, buf709, 73728, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten.convolution]
            buf661 = extern_kernels.convolution(buf659, buf660, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf661, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf659
            del buf660
            buf662 = reinterpret_tensor(buf617, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf617  # reuse
            # Topologically Sorted Source Nodes: [input_242, lif_forward_state_default_106], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf661, arg12_1, arg13_1, arg14_1, arg15_1, buf662, 1536, 169, stream=raw_stream0)
            del buf661
            # Topologically Sorted Source Nodes: [input_242, lif_forward_state_default_106], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf663 = torch.ops.snn_custom.lif_forward_state.default(buf662, buf616, 1.0, 0.0, 2.0, False)
            del buf616
            buf664 = buf663[0]
            assert_size_stride(buf664, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf664, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf665 = buf663[1]
            assert_size_stride(buf665, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf665, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf663
            buf666 = reinterpret_tensor(buf662, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf662  # reuse
            # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf664, buf666, 1536, 169, stream=raw_stream0)
            del buf664
            buf667 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            buf716 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_243, input_261], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(arg16_1, buf667, buf716, 98304, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
            buf668 = extern_kernels.convolution(buf666, buf667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf668, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf666
            del buf667
            buf669 = buf629; del buf629  # reuse
            # Topologically Sorted Source Nodes: [input_244, lif_forward_state_default_107], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf668, arg17_1, arg18_1, arg19_1, arg20_1, buf669, 1024, 169, stream=raw_stream0)
            del buf668
            # Topologically Sorted Source Nodes: [input_244, lif_forward_state_default_107], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf670 = torch.ops.snn_custom.lif_forward_state.default(buf669, buf623, 1.0, 0.0, 2.0, False)
            del buf623
            buf671 = buf670[0]
            assert_size_stride(buf671, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf671, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf672 = buf670[1]
            assert_size_stride(buf672, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf672, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf670
            buf673 = reinterpret_tensor(buf669, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf669  # reuse
            # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf671, buf673, 1024, 169, stream=raw_stream0)
            del buf671
            buf674 = buf625; del buf625  # reuse
            buf723 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_245, input_263], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf674, buf723, 65536, 9, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
            buf675 = extern_kernels.convolution(buf673, buf674, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf675, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf674
            buf676 = reinterpret_tensor(buf673, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf673  # reuse
            # Topologically Sorted Source Nodes: [input_246, lif_forward_state_default_108], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf675, arg22_1, arg23_1, arg24_1, arg25_1, buf676, 1024, 169, stream=raw_stream0)
            del buf675
            # Topologically Sorted Source Nodes: [input_246, lif_forward_state_default_108], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf677 = torch.ops.snn_custom.lif_forward_state.default(buf676, buf630, 1.0, 0.0, 2.0, False)
            del buf630
            del buf676
            buf678 = buf677[0]
            assert_size_stride(buf678, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf678, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf679 = buf677[1]
            assert_size_stride(buf679, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf679, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf677
            buf681 = buf632; del buf632  # reuse
            # Topologically Sorted Source Nodes: [input_247, x_26], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf678, buf681, 36864, stream=raw_stream0)
            del buf678
            buf682 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_26, x_27, input_249], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf681, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf682)
            del buf681
            # Topologically Sorted Source Nodes: [lif_forward_state_default_109], Original ATen: [snn_custom.lif_forward_state]
            buf683 = torch.ops.snn_custom.lif_forward_state.default(buf682, buf636, 1.0, 0.0, 2.0, False)
            del buf636
            buf684 = buf683[0]
            assert_size_stride(buf684, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf684, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf685 = buf683[1]
            assert_size_stride(buf685, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf685, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf683
            buf686 = buf682; del buf682  # reuse
            # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf684, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf686)
            del buf684
            # Topologically Sorted Source Nodes: [lif_forward_state_default_110], Original ATen: [snn_custom.lif_forward_state]
            buf687 = torch.ops.snn_custom.lif_forward_state.default(buf686, buf640, 1.0, 0.0, 2.0, False)
            del buf640
            del buf686
            buf688 = buf687[0]
            assert_size_stride(buf688, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf688, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf689 = buf687[1]
            assert_size_stride(buf689, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf689, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf687
            buf690 = buf641; del buf641  # reuse
            # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf688, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf690)
            del buf688
            # Topologically Sorted Source Nodes: [lif_forward_state_default_111], Original ATen: [snn_custom.lif_forward_state]
            buf691 = torch.ops.snn_custom.lif_forward_state.default(buf690, buf644, 1.0, 0.0, 2.0, False)
            del buf644
            del buf690
            buf692 = buf691[0]
            assert_size_stride(buf692, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf692, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf693 = buf691[1]
            assert_size_stride(buf693, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf693, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf691
            buf694 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_238, input_253], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_39.run(arg0_1, buf694, 12, 50176, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_238, input_253], Original ATen: [aten.select, aten.convolution]
            buf696 = extern_kernels.convolution(buf694, buf695, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf696, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf694
            del buf695
            buf697 = empty_strided_cuda((4, 64, 55, 55), (193600, 3025, 55, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_254, lif_forward_state_default_112], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf696, arg2_1, arg3_1, arg4_1, arg5_1, buf697, 256, 3025, stream=raw_stream0)
            del buf696
            # Topologically Sorted Source Nodes: [input_254, lif_forward_state_default_112], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf698 = torch.ops.snn_custom.lif_forward_state.default(buf697, buf651, 1.0, 0.0, 2.0, False)
            del buf651
            del buf697
            buf699 = buf698[0]
            assert_size_stride(buf699, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf699, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf700 = buf698[1]
            assert_size_stride(buf700, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf700, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf698
            buf701 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf699, buf701, 256, 729, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_256], Original ATen: [aten.convolution]
            buf703 = extern_kernels.convolution(buf701, buf702, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf703, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf701
            del buf702
            buf704 = empty_strided_cuda((4, 192, 27, 27), (139968, 729, 27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_257, lif_forward_state_default_113], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf703, arg7_1, arg8_1, arg9_1, arg10_1, buf704, 768, 729, stream=raw_stream0)
            del buf703
            # Topologically Sorted Source Nodes: [input_257, lif_forward_state_default_113], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf705 = torch.ops.snn_custom.lif_forward_state.default(buf704, buf658, 1.0, 0.0, 2.0, False)
            del buf658
            del buf704
            buf706 = buf705[0]
            assert_size_stride(buf706, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf706, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf707 = buf705[1]
            assert_size_stride(buf707, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf707, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf705
            buf708 = empty_strided_cuda((4, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_258], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf706, buf708, 768, 169, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_259], Original ATen: [aten.convolution]
            buf710 = extern_kernels.convolution(buf708, buf709, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf710, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            buf711 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_260, lif_forward_state_default_114], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf710, arg12_1, arg13_1, arg14_1, arg15_1, buf711, 1536, 169, stream=raw_stream0)
            del buf710
            # Topologically Sorted Source Nodes: [input_260, lif_forward_state_default_114], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf712 = torch.ops.snn_custom.lif_forward_state.default(buf711, buf665, 1.0, 0.0, 2.0, False)
            del buf665
            buf713 = buf712[0]
            assert_size_stride(buf713, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf713, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf714 = buf712[1]
            assert_size_stride(buf714, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf714, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf712
            buf715 = reinterpret_tensor(buf711, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf711  # reuse
            # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf713, buf715, 1536, 169, stream=raw_stream0)
            del buf713
            # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
            buf717 = extern_kernels.convolution(buf715, buf716, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf717, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf716
            buf718 = empty_strided_cuda((4, 256, 13, 13), (43264, 169, 13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_262, lif_forward_state_default_115], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf717, arg17_1, arg18_1, arg19_1, arg20_1, buf718, 1024, 169, stream=raw_stream0)
            del buf717
            # Topologically Sorted Source Nodes: [input_262, lif_forward_state_default_115], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf719 = torch.ops.snn_custom.lif_forward_state.default(buf718, buf672, 1.0, 0.0, 2.0, False)
            del buf672
            buf720 = buf719[0]
            assert_size_stride(buf720, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf720, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf721 = buf719[1]
            assert_size_stride(buf721, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf721, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf719
            buf722 = reinterpret_tensor(buf718, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf718  # reuse
            # Topologically Sorted Source Nodes: [input_263], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf720, buf722, 1024, 169, stream=raw_stream0)
            del buf720
            # Topologically Sorted Source Nodes: [input_263], Original ATen: [aten.convolution]
            buf724 = extern_kernels.convolution(buf722, buf723, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf724, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            buf725 = reinterpret_tensor(buf722, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf722  # reuse
            # Topologically Sorted Source Nodes: [input_264, lif_forward_state_default_116], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf724, arg22_1, arg23_1, arg24_1, arg25_1, buf725, 1024, 169, stream=raw_stream0)
            del buf724
            # Topologically Sorted Source Nodes: [input_264, lif_forward_state_default_116], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf726 = torch.ops.snn_custom.lif_forward_state.default(buf725, buf679, 1.0, 0.0, 2.0, False)
            del buf679
            del buf725
            buf727 = buf726[0]
            assert_size_stride(buf727, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf727, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf728 = buf726[1]
            assert_size_stride(buf728, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf728, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf726
            buf730 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_265, x_28], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_18.run(buf727, buf730, 36864, stream=raw_stream0)
            buf731 = empty_strided_cuda((4, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_28, x_29, input_267], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf730, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf731)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_117], Original ATen: [snn_custom.lif_forward_state]
            buf732 = torch.ops.snn_custom.lif_forward_state.default(buf731, buf685, 1.0, 0.0, 2.0, False)
            del buf685
            buf733 = buf732[0]
            assert_size_stride(buf733, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf733, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf734 = buf732[1]
            assert_size_stride(buf734, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf734, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf732
            buf735 = buf731; del buf731  # reuse
            # Topologically Sorted Source Nodes: [input_269], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf733, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf735)
            del buf733
            # Topologically Sorted Source Nodes: [lif_forward_state_default_118], Original ATen: [snn_custom.lif_forward_state]
            buf736 = torch.ops.snn_custom.lif_forward_state.default(buf735, buf689, 1.0, 0.0, 2.0, False)
            del buf689
            del buf735
            buf737 = buf736[0]
            assert_size_stride(buf737, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf737, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf738 = buf736[1]
            assert_size_stride(buf738, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf738, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf736
            buf739 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_270], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf737, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf739)
            # Topologically Sorted Source Nodes: [lif_forward_state_default_119], Original ATen: [snn_custom.lif_forward_state]
            buf740 = torch.ops.snn_custom.lif_forward_state.default(buf739, buf693, 1.0, 0.0, 2.0, False)
            del buf693
            buf741 = buf740[0]
            assert_size_stride(buf741, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf741, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf742 = buf740[1]
            assert_size_stride(buf742, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf742, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf740
            buf743 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_255, input_271], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_40.run(arg0_1, buf743, 12, 50176, stream=raw_stream0)
            del arg0_1
            buf744 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_255, input_271], Original ATen: [aten.select, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_select_1.run(arg1_1, buf744, 192, 121, stream=raw_stream0)
            del arg1_1
            # Topologically Sorted Source Nodes: [getitem_255, input_271], Original ATen: [aten.select, aten.convolution]
            buf745 = extern_kernels.convolution(buf743, buf744, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf745, (4, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf743
            del buf744
            buf746 = buf699; del buf699  # reuse
            # Topologically Sorted Source Nodes: [input_272, lif_forward_state_default_120], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_2.run(buf745, arg2_1, arg3_1, arg4_1, arg5_1, buf746, 256, 3025, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del buf745
            # Topologically Sorted Source Nodes: [input_272, lif_forward_state_default_120], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf747 = torch.ops.snn_custom.lif_forward_state.default(buf746, buf700, 1.0, 0.0, 2.0, False)
            del buf700
            del buf746
            buf748 = buf747[0]
            assert_size_stride(buf748, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf748, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf749 = buf747[1]
            assert_size_stride(buf749, (4, 64, 55, 55), (193600, 3025, 55, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf749, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf747
            buf750 = empty_strided_cuda((4, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_4.run(buf748, buf750, 256, 729, stream=raw_stream0)
            del buf748
            buf751 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_274], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_5.run(arg6_1, buf751, 12288, 25, stream=raw_stream0)
            del arg6_1
            # Topologically Sorted Source Nodes: [input_274], Original ATen: [aten.convolution]
            buf752 = extern_kernels.convolution(buf750, buf751, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf752, (4, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf750
            del buf751
            buf753 = buf706; del buf706  # reuse
            # Topologically Sorted Source Nodes: [input_275, lif_forward_state_default_121], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_6.run(buf752, arg7_1, arg8_1, arg9_1, arg10_1, buf753, 768, 729, stream=raw_stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            del buf752
            # Topologically Sorted Source Nodes: [input_275, lif_forward_state_default_121], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf754 = torch.ops.snn_custom.lif_forward_state.default(buf753, buf707, 1.0, 0.0, 2.0, False)
            del buf707
            del buf753
            buf755 = buf754[0]
            assert_size_stride(buf755, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf755, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf756 = buf754[1]
            assert_size_stride(buf756, (4, 192, 27, 27), (139968, 729, 27, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf756, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf754
            buf757 = buf708; del buf708  # reuse
            # Topologically Sorted Source Nodes: [input_276], Original ATen: [aten.max_pool2d_with_indices]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_8.run(buf755, buf757, 768, 169, stream=raw_stream0)
            del buf755
            buf758 = buf709; del buf709  # reuse
            # Topologically Sorted Source Nodes: [input_277], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_9.run(arg11_1, buf758, 73728, 9, stream=raw_stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [input_277], Original ATen: [aten.convolution]
            buf759 = extern_kernels.convolution(buf757, buf758, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf759, (4, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf757
            del buf758
            buf760 = reinterpret_tensor(buf715, (4, 384, 13, 13), (64896, 169, 13, 1), 0); del buf715  # reuse
            # Topologically Sorted Source Nodes: [input_278, lif_forward_state_default_122], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_10.run(buf759, arg12_1, arg13_1, arg14_1, arg15_1, buf760, 1536, 169, stream=raw_stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            del buf759
            # Topologically Sorted Source Nodes: [input_278, lif_forward_state_default_122], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf761 = torch.ops.snn_custom.lif_forward_state.default(buf760, buf714, 1.0, 0.0, 2.0, False)
            del buf714
            buf762 = buf761[0]
            assert_size_stride(buf762, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf762, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf763 = buf761[1]
            assert_size_stride(buf763, (4, 384, 13, 13), (64896, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf763, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf761
            buf764 = reinterpret_tensor(buf760, (4, 384, 13, 13), (64896, 1, 4992, 384), 0); del buf760  # reuse
            # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(buf762, buf764, 1536, 169, stream=raw_stream0)
            del buf762
            buf765 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_13.run(arg16_1, buf765, 98304, 9, stream=raw_stream0)
            del arg16_1
            # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
            buf766 = extern_kernels.convolution(buf764, buf765, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf766, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf764
            del buf765
            buf767 = buf727; del buf727  # reuse
            # Topologically Sorted Source Nodes: [input_280, lif_forward_state_default_123], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf766, arg17_1, arg18_1, arg19_1, arg20_1, buf767, 1024, 169, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            del buf766
            # Topologically Sorted Source Nodes: [input_280, lif_forward_state_default_123], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf768 = torch.ops.snn_custom.lif_forward_state.default(buf767, buf721, 1.0, 0.0, 2.0, False)
            del buf721
            buf769 = buf768[0]
            assert_size_stride(buf769, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf769, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf770 = buf768[1]
            assert_size_stride(buf770, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf770, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf768
            buf771 = reinterpret_tensor(buf767, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf767  # reuse
            # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_16.run(buf769, buf771, 1024, 169, stream=raw_stream0)
            del buf769
            buf772 = buf723; del buf723  # reuse
            # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_17.run(arg21_1, buf772, 65536, 9, stream=raw_stream0)
            del arg21_1
            # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
            buf773 = extern_kernels.convolution(buf771, buf772, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf773, (4, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf772
            buf774 = reinterpret_tensor(buf771, (4, 256, 13, 13), (43264, 169, 13, 1), 0); del buf771  # reuse
            # Topologically Sorted Source Nodes: [input_282, lif_forward_state_default_124], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_lif_forward_state_zeros_like_14.run(buf773, arg22_1, arg23_1, arg24_1, arg25_1, buf774, 1024, 169, stream=raw_stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del buf773
            # Topologically Sorted Source Nodes: [input_282, lif_forward_state_default_124], Original ATen: [aten._native_batch_norm_legit_no_training, snn_custom.lif_forward_state]
            buf775 = torch.ops.snn_custom.lif_forward_state.default(buf774, buf728, 1.0, 0.0, 2.0, False)
            del buf728
            del buf774
            buf776 = buf775[0]
            assert_size_stride(buf776, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf776, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf777 = buf775[1]
            assert_size_stride(buf777, (4, 256, 13, 13), (43264, 169, 13, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf777, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf775
            buf778 = buf730; del buf730  # reuse
            buf779 = buf778; del buf778  # reuse
            # Topologically Sorted Source Nodes: [input_283, x_30], Original ATen: [aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_max_pool2d_with_indices_41.run(buf779, buf776, 36864, stream=raw_stream0)
            del buf776
            buf780 = buf737; del buf737  # reuse
            # Topologically Sorted Source Nodes: [x_30, x_31, input_285], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf779, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg26_1, (9216, 4096), (1, 9216), 0), out=buf780)
            del arg26_1
            del buf779
            # Topologically Sorted Source Nodes: [lif_forward_state_default_125], Original ATen: [snn_custom.lif_forward_state]
            buf781 = torch.ops.snn_custom.lif_forward_state.default(buf780, buf734, 1.0, 0.0, 2.0, False)
            del buf734
            buf782 = buf781[0]
            assert_size_stride(buf782, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf782, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf783 = buf781[1]
            assert_size_stride(buf783, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf783, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf781
            buf784 = buf780; del buf780  # reuse
            # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf782, reinterpret_tensor(arg27_1, (4096, 4096), (1, 4096), 0), out=buf784)
            del arg27_1
            del buf782
            # Topologically Sorted Source Nodes: [lif_forward_state_default_126], Original ATen: [snn_custom.lif_forward_state]
            buf785 = torch.ops.snn_custom.lif_forward_state.default(buf784, buf738, 1.0, 0.0, 2.0, False)
            del buf738
            del buf784
            buf786 = buf785[0]
            assert_size_stride(buf786, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf786, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf787 = buf785[1]
            assert_size_stride(buf787, (4, 4096), (4096, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf787, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf785
            buf788 = buf739; del buf739  # reuse
            # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf786, reinterpret_tensor(arg28_1, (4096, 10), (1, 4096), 0), out=buf788)
            del arg28_1
            del buf786
            # Topologically Sorted Source Nodes: [lif_forward_state_default_127], Original ATen: [snn_custom.lif_forward_state]
            buf789 = torch.ops.snn_custom.lif_forward_state.default(buf788, buf742, 1.0, 0.0, 2.0, False)
            del buf742
            del buf788
            buf790 = buf789[0]
            assert_size_stride(buf790, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf790, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            buf791 = buf789[1]
            assert_size_stride(buf791, (4, 10), (10, 1), 'torch.ops.snn_custom.lif_forward_state.default')
            assert_alignment(buf791, 16, 'torch.ops.snn_custom.lif_forward_state.default')
            del buf789
            buf792 = buf55; del buf55  # reuse
            buf793 = buf792; del buf792  # reuse
            # Topologically Sorted Source Nodes: [out_spikes_counter, out_spikes_counter_1, out_spikes_counter_2, out_spikes_counter_3, out_spikes_counter_4, out_spikes_counter_5, out_spikes_counter_6, out_spikes_counter_7, out_spikes_counter_8, out_spikes_counter_9, out_spikes_counter_10, out_spikes_counter_11, out_spikes_counter_12, out_spikes_counter_13, out_spikes_counter_14, out_spikes_counter_15, truediv], Original ATen: [aten.add, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_42.run(buf793, buf104, buf153, buf202, buf251, buf300, buf349, buf398, buf447, buf496, buf545, buf594, buf643, buf692, buf741, buf790, 40, stream=raw_stream0)
            del buf104
            del buf153
            del buf202
            del buf251
            del buf300
            del buf349
            del buf398
            del buf447
            del buf496
            del buf545
            del buf594
            del buf643
            del buf692
            del buf741
            del buf790
        return (buf793, buf749, buf756, buf763, buf770, buf777, buf783, buf787, buf791, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16, 4, 3, 224, 224), (602112, 150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((4096, 9216), (9216, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((10, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
