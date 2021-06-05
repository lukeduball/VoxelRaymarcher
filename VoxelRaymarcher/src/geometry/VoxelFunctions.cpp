#include "VoxelFunctions.h"

#include <assert.h>

uint32_t VoxelFunctions::generate3DPoint(uint32_t x, uint32_t y, uint32_t z)
{
	//Ensure the x, y, and z coordinates are all less than 1024 because each coordinate only gets 10 bits
	assert(x < 1024 && y < 1024 && z < 1024);
	return (x << 20) | (y << 10) | z;
}

uint32_t VoxelFunctions::generateRGBColor(uint32_t r, uint32_t g, uint32_t b)
{
	assert(r < 256 && g < 256 && b < 256);
	return (r << 16) | (g << 8) | b;
}
