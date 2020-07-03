#include "control.h"
#include "operator.cuh"

__device__ __forceinline__
uint Phase_t::GetKernelShape(uint d) const 	{ return __ldg(&kernel_shapes[d]); }

__device__ __forceinline__
float4 Phase_t::GetBias_v2(uint k) const 			{ return __ldg(&bias_v2[k]); }

__device__ __forceinline__
float4 Phase_t::GetKernel_v2(uint k) const 			{ return __ldg(&kernel_v2[k]); }

__device__ __forceinline__
float4 Phase_t::GetOffset_v2(uint k) const 			{ return __ldg(&beta_v2[k]); }

__device__ __forceinline__
float4 Phase_t::GetScale_v2(uint k) const 			{ return __ldg(&gamma_v2[k]); }

