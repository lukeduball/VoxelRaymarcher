#pragma once

#include <unordered_map>
#include <vector>

#include "VoxelFunctions.cuh"

#include "../memory/MemoryUtils.h"

#include "../storage/VoxelClusterStore.cuh"
#include "../storage/CuckooHashTable.cuh"

class VoxelSceneCPU
{
public:
	void insertVoxel(int32_t x, int32_t y, int32_t z, uint32_t color)
	{
		//Find the region ID
		int32_t voxelStructureIDX = std::floorf(x / (float)BLOCK_SIZE);
		int32_t voxelStructureIDY = std::floorf(y / (float)BLOCK_SIZE);
		int32_t voxelStructureIDZ = std::floorf(z / (float)BLOCK_SIZE);
		
		//To account for negative values, add the BLOCK_SIZE to the remainder and find the remainer again
		uint32_t localVoxelX = ((x % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;
		uint32_t localVoxelY = ((y % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;
		uint32_t localVoxelZ = ((z % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;

		int32_t minVoxelRegionCoord = std::min(voxelStructureIDX, std::min(voxelStructureIDY, voxelStructureIDZ));
		int32_t maxVoxelRegionCoord = std::max(voxelStructureIDX, std::max(voxelStructureIDY, voxelStructureIDZ));

		//Set the minimum and maximum coordinates if they are larger then the previous min and max
		if (minVoxelRegionCoord < minCoord)
			minCoord = minVoxelRegionCoord;
		if (maxVoxelRegionCoord > maxCoord)
			maxCoord = maxVoxelRegionCoord;


		//Check if the region is already added to the map and create the map for that region if it has not
		Vector3i storageKey = Vector3i(voxelStructureIDX, voxelStructureIDY, voxelStructureIDZ);
		if (voxelSceneStorage.find(storageKey) == voxelSceneStorage.end())
		{
			voxelSceneStorage[storageKey] = std::unordered_map<uint32_t, uint32_t>();
		}
		//Place the voxel in the voxel map of the region
		voxelSceneStorage[storageKey][voxelfunc::generate3DPoint(localVoxelX, localVoxelY, localVoxelZ)] = color;
	}

	//Generate the voxel scene and copy it to the GPU
	void generateVoxelScene()
	{
		int32_t arrayDiameter = maxCoord - minCoord + 1;
		uint32_t arraySize = arrayDiameter * arrayDiameter * arrayDiameter;

		std::cout << "There are : " << voxelSceneStorage.size() << "/" << arraySize << " regions that are filled" << std::endl;

		VoxelClusterStore** hostPtrArray = static_cast<VoxelClusterStore**>(malloc(sizeof(VoxelClusterStore*) * arraySize));
		std::memset(hostPtrArray, 0, sizeof(VoxelClusterStore*) * arraySize);

		for (std::pair<Vector3i, std::unordered_map<uint32_t, uint32_t>> pair : voxelSceneStorage)
		{
			Vector3i arrayLocation = pair.first - Vector3i(minCoord, minCoord, minCoord);
			uint32_t arrayIndex = arrayLocation.getX() + arrayLocation.getY() * arrayDiameter + arrayLocation.getZ() * arrayDiameter * arrayDiameter;

			//Generate the voxel cluster store on the CPU
			VoxelClusterStore* voxelClusterStore = new VoxelClusterStore(pair.second);
			VoxelClusterStore* devicePtr;
			//Move the voxel cluster store to the GPU
			cudaMalloc(&devicePtr, sizeof(VoxelClusterStore));
			cudaMemcpy(devicePtr, voxelClusterStore, sizeof(VoxelClusterStore), cudaMemcpyHostToDevice);
			hostPtrArray[arrayIndex] = devicePtr;
		}
		std::cout << "Storage Structures Generated" << std::endl;

		//Copy the region table to the GPU
		cudaMalloc(&deviceVoxelScene, sizeof(VoxelClusterStore*) * arraySize);
		cudaMemcpy(deviceVoxelScene, hostPtrArray, sizeof(VoxelClusterStore*) * arraySize, cudaMemcpyHostToDevice);

		free(hostPtrArray);
	}

	void cleanupVoxelScene()
	{
		uint32_t arraySize = getArraySize();
		for (uint32_t i = 0; i < arraySize; i++)
		{
			//TODO add host tracker to clean up this memory
			//cudaFree(deviceVoxelScene[i]);
		}
		CudaMemoryUtils::ManagedCudaFree(deviceVoxelScene, "Voxel Scene (raw table of pointers to the regions' storage structures)");
		cudaFree(deviceVoxelScene);
	}

	uint32_t getArrayDiameter()
	{
		return maxCoord - minCoord + 1;
	}

	uint32_t getArraySize()
	{
		int32_t arrayDiameter = getArrayDiameter();
		return arrayDiameter * arrayDiameter * arrayDiameter;
	}

	int32_t getMinCoord()
	{
		return minCoord;
	}

	VoxelClusterStore** deviceVoxelScene;

private:
	//Map will be converted to an array of VoxelStructures which contain the actual voxels
	std::unordered_map<Vector3i, std::unordered_map<uint32_t, uint32_t>, Vector3iHashFunction> voxelSceneStorage;
	//Keeps track of the minumum and maximum coordinates to calculate the radius to allocate
	int32_t minCoord = 0;
	int32_t maxCoord = 0;
};
