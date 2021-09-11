#pragma once

#include <cuda_runtime.h>

#include <assert.h>
#include <stdint.h>

constexpr float EPSILON = 0.0001f;
__constant__ const uint32_t EMPTY_KEY = 1 << 30;
__constant__ const uint32_t EMPTY_VAL = 1 << 30;
__constant__ const uint32_t FINISH_VAL = EMPTY_VAL + 1;
__constant__ const uint32_t BLOCK_SIZE = 64;

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
}