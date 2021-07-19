﻿#pragma once

#include <algorithm>
#include <stdint.h>
#include <unordered_map>

#include "../geometry/VoxelFunctions.cuh"

__constant__ const uint32_t EMPTY_VAL = 1 << 30;

struct VoxelPair
{
	uint32_t key;
	uint32_t value;
};

__host__ bool compareVoxelPair(const VoxelPair& v1, const VoxelPair& v2)
{
	return(v1.key < v2.key);
}

__device__ short getVoxelClusterID(uint32_t x, uint32_t y, uint32_t z)
{
	return ((x / 8) << 6) | ((y / 8) << 3) | (z / 8);
}

__host__ short getVoxelClusterID(uint32_t voxelID)
{
	uint32_t x = voxelID >> 20;
	uint32_t y = (voxelID >> 10) & 0x3FF;
	uint32_t z = voxelID & 0x3FF;
	return ((x / 8) << 6) | ((y / 8) << 3) | (z / 8);
}

class VoxelClusterStore
{
public:
	__host__ VoxelClusterStore(const std::unordered_map<uint32_t, uint32_t>& existingTable)
	{
		uint32_t* blockMemAddress[512] = { nullptr };
		std::vector<std::vector<VoxelPair>> blockSets = std::vector<std::vector<VoxelPair>>(512);
		//Iterate through the elements in the map and place them in their cooresponding set
		for (const auto& pair : existingTable)
		{
			short clusterID = getVoxelClusterID(pair.first);
			blockSets[clusterID].push_back(VoxelPair{ pair.first, pair.second });
		}
		//Allocate all the memory necessary for this VoxelClusterStore
		size_t memorySize = sizeof(VoxelPair) * existingTable.size() + 512 * sizeof(uint32_t);

		//Allocate space on the GPU for the voxel block memory
		cudaMalloc(&deviceBlocksMemory, memorySize);

		//Allocate space on the GPU for the voxel memory address array
		cudaMalloc(&deviceBlockMemAddress, sizeof(uint32_t*) * 512);

		//Allocate memory on the CPU to populate the blocks so that it can be copied to the GPU later
		uint32_t* blocksMemory = static_cast<uint32_t*>(malloc(memorySize));
		uint32_t* lastMemoryAddress = blocksMemory;
		uint32_t* lastDeviceBlockMemoryAddress = deviceBlocksMemory;
		//Go through each set and sort the items
		for (uint32_t i = 0; i < 512; i++)
		{
			//Do not do any processing if the block is empty
			if (blockSets[i].empty())
				continue;
			std::sort(blockSets[i].begin(), blockSets[i].end(), compareVoxelPair);
			uint32_t blockSize = static_cast<uint32_t>(blockSets[i].size());
			blockMemAddress[i] = lastDeviceBlockMemoryAddress;
			//Set the first value in the block as the size of the block
			*lastMemoryAddress = blockSize;
			//Copy the rest of the contents into memory
			memcpy(lastMemoryAddress + 1, blockSets[i].data(), sizeof(VoxelPair) * blockSize);
			//Increment both the CPU lastMemoryAddress and the GPU location lastDeviceBlockMemoryAddress
			lastMemoryAddress = lastMemoryAddress + (blockSize*2 + 1);
			lastDeviceBlockMemoryAddress = lastDeviceBlockMemoryAddress + (blockSize * 2 + 1);
		}

		//Copy the voxel block memory contents to the GPU
		cudaMemcpy(deviceBlocksMemory, blocksMemory, memorySize, cudaMemcpyHostToDevice);

		//Copy the voxel block memory contents to the GPU
		cudaMemcpy(deviceBlockMemAddress, blockMemAddress, sizeof(uint32_t*) * 512, cudaMemcpyHostToDevice);

		free(blocksMemory);
	}

	__host__ ~VoxelClusterStore()
	{
		cudaFree(deviceBlocksMemory);
		cudaFree(deviceBlockMemAddress);
	}

	__device__ uint32_t getColorFromVoxel(uint32_t x, uint32_t y, uint32_t z)
	{
		uint32_t voxelID = voxelfunc::generate3DPoint(x, y, z);
		short clusterID = getVoxelClusterID(x, y, z);
		uint32_t* blockLocation = deviceBlockMemAddress[clusterID];
		if (blockLocation != nullptr)
		{
			//Binary Search
			uint32_t blockSize = *blockLocation;
			int32_t low = 0;
			int32_t high = blockSize-1;
			while (low <= high)
			{
				int32_t mid = low + (high - low) / 2;
				//Check if the key matches
				if (blockLocation[mid*2 + 1] == voxelID)
				{
					//Return the value pairing
					return blockLocation[mid*2 + 2];
				}

				if (blockLocation[mid*2 + 1] < voxelID)
				{
					low = mid + 1;
				}
				else
				{
					high = mid - 1;
				}
			}
		}
		return EMPTY_VAL;
	}

private:
	uint32_t* deviceBlocksMemory;
	uint32_t** deviceBlockMemAddress;
};