#pragma once

#include <unordered_map>

class VoxelCube
{
public:
	static void generateVoxelCube(std::unordered_map<uint32_t, uint32_t>& voxelTable, uint32_t x, uint32_t y, uint32_t z, uint32_t halfWidth);
};