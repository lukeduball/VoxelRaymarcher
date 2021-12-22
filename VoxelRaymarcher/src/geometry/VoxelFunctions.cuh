#pragma once

#include <cuda_runtime.h>

#include <assert.h>
#include <stdint.h>
#include <cstdlib>
#include <cstdio>

#include "../math/Vector3.cuh"

#ifndef NDEBUG
#define ASSERT(cond, str) if(!(cond)) {printf("ASSERTION FAILED: (%s) %s in File %s, Line %d\n", #cond, str, __FILE__, __LINE__); asm("trap;"); }
#else
#define ASSERT(cond, str)
#endif // NDEBUG


constexpr float EPSILON = 0.0001f;
__constant__ const uint32_t EMPTY_KEY = 1 << 30;
__constant__ const uint32_t EMPTY_VAL = 1 << 30;
__constant__ const int32_t BLOCK_SIZE = 64;
__constant__ const int32_t CLUSTER_SIZE = 8;

__constant__ Vector3f LIGHT_DIRECTION;
__constant__ Vector3f LIGHT_COLOR;
__constant__ Vector3f LIGHT_POSITION;
__constant__ const float LIGHT_CONSTANT = 1.0f;
__constant__ const float LIGHT_LINEAR = 0.045f;
__constant__ const float LIGHT_QUADRATIC = 0.0075f;

__constant__ bool USE_POINT_LIGHT = false;
__constant__ bool USE_SHADOWS = false;

namespace voxelfunc
{
	__host__ __device__ uint32_t generate3DPoint(uint32_t x, uint32_t y, uint32_t z)
	{
		//Ensure the x, y, and z coordinates are all less than 1024 because each coordinate only gets 10 bits
		assert(x < 1024 && y < 1024 && z < 1024);
		return (x << 20) | (y << 10) | z;
	}

	__host__ __device__ uint32_t generateRGBColor(uint32_t r, uint32_t g, uint32_t b)
	{
		assert(r < 256 && g < 256 && b < 256);
		return (r << 16) | (g << 8) | b;
	}

	__host__ __device__ __forceinline__ uint32_t getRedComponent(uint32_t color)
	{
		return color >> 16;
	}

	__host__ __device__ __forceinline__ uint32_t getGreenComponent(uint32_t color)
	{
		return (color >> 8) & 0xFF;
	}

	__host__ __device__ __forceinline__ uint32_t getBlueComponent(uint32_t color)
	{
		return color & 0xFF;
	}

	__host__ __device__ __forceinline__ Vector3f convertRGBIntegerColorToVector(uint32_t color)
	{
		float red = getRedComponent(color) / 255.0f;
		float green = getGreenComponent(color) / 255.0f;
		float blue = getBlueComponent(color) / 255.0f;
		return Vector3f(red, green, blue);
	}

	__host__ __device__ __forceinline__ uint32_t convertRGBVectorToInteger(const Vector3f& colorVec)
	{
		uint32_t r = static_cast<uint32_t>(colorVec.getX() * 255.0f);
		uint32_t g = static_cast<uint32_t>(colorVec.getY() * 255.0f);
		uint32_t b = static_cast<uint32_t>(colorVec.getZ() * 255.0f);
		return generateRGBColor(r, g, b);
	}

	__host__ __device__ __forceinline__ float floorFloatWithIntegers(float f)
	{
		return static_cast<float>(static_cast<int32_t>(f));
	}

	__host__ __device__ __forceinline__ float ceilFloatWithIntegers(float f)
	{
		return static_cast<float>(static_cast<int32_t>(f) + 1);
	}
}