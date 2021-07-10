#pragma once

#include <algorithm>
#include <stdint.h>
#include <unordered_map>

#include "../geometry/VoxelFunctions.h"

const uint32_t EMPTY_VAL = 1 << 30;

//This file will be used to implement the Voxel Cluster Storage. This will cluster voxels into NxNxN blocks.
//To access a specific voxel, its block is determined and then a binary search will be done on the clustered voxels to find the result
//Special cases --> EMPTY BLOCKS (no voxels containted inside)

//Block which holds the data
//Control Block which contains the info on where to find the data

struct VoxelPair
{
	uint32_t key;
	uint32_t value;
};

bool compareVoxelPair(const VoxelPair& v1, const VoxelPair& v2)
{
	return(v1.key < v2.key);
}

short getVoxelClusterID(char x, char y, char z)
{
	return ((x / 8) << 6) | ((y / 8) << 3) | (z / 8);
}

short getVoxelClusterID(uint32_t voxelID)
{
	uint32_t x = voxelID >> 20;
	uint32_t y = (voxelID >> 10) & 0x3FF;
	uint32_t z = voxelID & 0x3FF;
	return ((x / 8) << 6) | ((y / 8) << 3) | (z / 8);
}

class VoxelCluster
{
public:
	short identifier;

	inline char getVoxelClusterX()
	{
		return identifier >> 16;
	}

	inline char getVoxelClusterY()
	{
		return (identifier >> 8) & 0xF;
	}

	inline char getVoxelClusterZ()
	{
		return identifier & 0xF;
	}
};

class VoxelClusterStore
{
public:
	VoxelClusterStore(const std::unordered_map<uint32_t, uint32_t>& existingTable) : blockMemAddress()
	{
		std::vector<std::vector<VoxelPair>> blockSets = std::vector<std::vector<VoxelPair>>(512);
		//Iterate through the elements in the map and place them in their cooresponding set
		for (const auto& pair : existingTable)
		{
			short clusterID = getVoxelClusterID(pair.first);
			blockSets[clusterID].push_back(VoxelPair{ pair.first, pair.second });
		}
		//Allocate all the memory necessary for this VoxelClusterStore
		blocksMemory = static_cast<uint32_t*>(malloc(sizeof(VoxelPair) * existingTable.size() + 512 * sizeof(uint32_t)));
		uint32_t* lastMemoryAddress = blocksMemory;
		//Go through each set and sort the items
		for (uint32_t i = 0; i < 512; i++)
		{
			//Do not do any processing if the block is empty
			if (blockSets[i].empty())
				continue;
			std::sort(blockSets[i].begin(), blockSets[i].end(), compareVoxelPair);
			uint32_t blockSize = static_cast<uint32_t>(blockSets[i].size());
			blockMemAddress[i] = lastMemoryAddress;
			//Set the first value in the block as the size of the block
			*lastMemoryAddress = blockSize;
			//Copy the rest of the contents into memory
			memcpy(lastMemoryAddress + 1, blockSets[i].data(), sizeof(VoxelPair) * blockSize);
			lastMemoryAddress = lastMemoryAddress + (blockSize*2 + 1);
		}		
	}

	~VoxelClusterStore()
	{
		free(blocksMemory);
	}

	uint32_t* blocksMemory;
	uint32_t* blockMemAddress[512];

	uint32_t getColorFromVoxel(char x, char y, char z)
	{
		uint32_t voxelID = VoxelFunctions::generate3DPoint(x, y, z);
		short clusterID = getVoxelClusterID(x, y, z);
		uint32_t* blockLocation = blockMemAddress[clusterID];
		if (blockLocation == nullptr)
		{
			return EMPTY_VAL;
		}
		else
		{
			//Binary Search
			uint32_t blockSize = *blockLocation;
			uint32_t low = 0;
			uint32_t high = blockSize-1;
			while (low <= high)
			{
				uint32_t mid = low + (high - low) / 2;
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

			return EMPTY_VAL;
		}
	}
};