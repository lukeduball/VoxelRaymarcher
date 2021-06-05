#pragma once

#include <stdint.h>

struct VoxelFunctions
{
	static uint32_t generate3DPoint(uint32_t x, uint32_t y, uint32_t z);
	static uint32_t generateRGBColor(uint32_t r, uint32_t g, uint32_t b);
};