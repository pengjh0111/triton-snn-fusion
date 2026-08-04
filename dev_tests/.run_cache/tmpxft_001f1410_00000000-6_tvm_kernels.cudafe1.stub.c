#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "tmpxft_001f1410_00000000-3_tvm_kernels.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z13main_kernel_3PfS_PaS_S_S_S_(float *__restrict__, float *__restrict__, signed char *__restrict__, float *__restrict__, float *__restrict__, float *__restrict__, float *__restrict__);
extern __attribute__((visibility("hidden"))) void __device_stub__Z11main_kernelPfS_(float *__restrict__, float *__restrict__);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13main_kernel_1PfS_S_(float *__restrict__, float *__restrict__, float *__restrict__);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13main_kernel_2PfS_(float *__restrict__, float *__restrict__);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z13main_kernel_3PfS_PaS_S_S_S_(float *__restrict__ __par0, float *__restrict__ __par1, signed char *__restrict__ __par2, float *__restrict__ __par3, float *__restrict__ __par4, float *__restrict__ __par5, float *__restrict__ __par6){ float *__T0;
 float *__T1;
 signed char *__T2;
 float *__T3;
 float *__T4;
 float *__T5;
 float *__T6;
__cudaLaunchPrologue(7);__T0 = __par0;__cudaSetupArgSimple(__T0, 0UL);__T1 = __par1;__cudaSetupArgSimple(__T1, 8UL);__T2 = __par2;__cudaSetupArgSimple(__T2, 16UL);__T3 = __par3;__cudaSetupArgSimple(__T3, 24UL);__T4 = __par4;__cudaSetupArgSimple(__T4, 32UL);__T5 = __par5;__cudaSetupArgSimple(__T5, 40UL);__T6 = __par6;__cudaSetupArgSimple(__T6, 48UL);__cudaLaunch(((char *)((void ( *)(float *__restrict__, float *__restrict__, signed char *__restrict__, float *__restrict__, float *__restrict__, float *__restrict__, float *__restrict__))main_kernel_3)));}
# 26 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
void main_kernel_3( float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,signed char *__restrict__ __cuda_2,float *__restrict__ __cuda_3,float *__restrict__ __cuda_4,float *__restrict__ __cuda_5,float *__restrict__ __cuda_6)
# 26 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
{__device_stub__Z13main_kernel_3PfS_PaS_S_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 44 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
}
# 1 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpxft_001f1410_00000000-6_tvm_kernels.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z11main_kernelPfS_( float *__restrict__ __par0,  float *__restrict__ __par1) {  float *__T7;
 float *__T8;
__cudaLaunchPrologue(2); __T7 = __par0; __cudaSetupArgSimple(__T7, 0UL); __T8 = __par1; __cudaSetupArgSimple(__T8, 8UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, float *__restrict__))main_kernel))); }
# 46 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
void main_kernel( float *__restrict__ __cuda_0,float *__restrict__ __cuda_1)
# 46 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
{__device_stub__Z11main_kernelPfS_( __cuda_0,__cuda_1);
# 233 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpi700vtwz/tvm_kernels.cu"
}
# 1 "/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/tmpxft_001f1410_00000000-6_tvm_kernels.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13main_kernel_1PfS_S_( float *__restrict__ __par0,  float *__restrict__ __par1,  float *__restrict__ __par2) {  float *__T9;
 float *__T10;
 float *__T11;
__cudaLaunchPrologue(3); __T9 = __par0; __cudaSetupArgSim