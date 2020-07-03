#pragma once

#include <unistd.h>
#include <limits.h>

#include "cuda.h"
#include "cuda_runtime.h"
#ifdef __CUDACC__
#include "device_launch_parameters.h"
#endif

using namespace std;

__host__ __device__ __forceinline__ void operator+=(float4& a, const float4 b)
{ 
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; 
}
__host__ __device__ __forceinline__ void operator+=(float4& a, const float b)
{
	a.x += b; a.y += b; a.z += b; a.w += b; 
}
__host__ __device__ __forceinline__ void operator-=(float4& a, const float4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
__host__ __device__ __forceinline__ void operator-=(float4& a, const float b)
{
	a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}
__host__ __device__ __forceinline__ void operator*=(float4& a, const float4 b)
{
	a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
__host__ __device__ __forceinline__ void operator*=(float4& a, const float b)
{
	a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}
__host__ __device__ __forceinline__ void operator/=(float4& a, const float4 b)
{
	a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
__host__ __device__ __forceinline__ void operator/=(float4& a, const float b)
{
	a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}
__host__ __device__ __forceinline__ float4 operator+(const float4 a, const float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ __forceinline__ float4 operator+(const float4 a, const float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__host__ __device__ __forceinline__ float4 operator+(const float a, const float4 b)
{
	return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}
__host__ __device__ __forceinline__ float4 operator-(const float4 a, const float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ __forceinline__ float4 operator-(const float4 a, const float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__host__ __device__ __forceinline__ float4 operator-(const float a, const float4 b)
{
	return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}
__host__ __device__ __forceinline__ float4 operator*(const float4 a, const float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ __forceinline__ float4 operator*(const float4 a, const float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__host__ __device__ __forceinline__ float4 operator*(const float a, const float4 b)
{
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}
__host__ __device__ __forceinline__ float4 operator/(const float4 a, const float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__host__ __device__ __forceinline__ float4 operator/(const float4 a, const float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__host__ __device__ __forceinline__ float4 operator/(const float a, const float4 b)
{
	return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}

__host__ __device__ __forceinline__ float sum(const float4 a)
{
	return (a.x + a.y + a.z + a.w);
}

__device__ __forceinline__ float4 __fsqrt_rn(const float4 a)
{
	return make_float4(__fsqrt_rn(a.x), __fsqrt_rn(a.y), __fsqrt_rn(a.z), __fsqrt_rn(a.w));
}

__device__ __forceinline__ float dot(const float4 a, const float4 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}

__device__ __forceinline__ float4 tanhf(const float4 a)
{
	return make_float4(tanhf(a.x), tanhf(a.y), tanhf(a.z), tanhf(a.w));
}
