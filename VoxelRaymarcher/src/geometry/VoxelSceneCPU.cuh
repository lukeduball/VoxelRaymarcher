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

		minCoords[0] = std::min(voxelStructureIDX, minCoords[0]);
		minCoords[1] = std::min(voxelStructureIDY, minCoords[1]);
		minCoords[2] = std::min(voxelStructureIDZ, minCoords[2]);
		
		maxCoords[0] = std::max(voxelStructureIDX, maxCoords[0]);
		maxCoords[1] = std::max(voxelStructureIDY, maxCoords[1]);
		maxCoords[2] = std::max(voxelStructureIDZ, maxCoords[2]);


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
		Vector3i arrayDiameter = getArrayDiameter();
		uint32_t arraySize = arrayDiameter.getX() * arrayDiameter.getY() * arrayDiameter.getZ();

		std::cout << "There are : " << voxelSceneStorage.size() << "/" << arraySize << " regions that are filled" << std::endl;

		VoxelClusterStore** hostPtrArray = static_cast<VoxelClusterStore**>(malloc(sizeof(VoxelClusterStore*) * arraySize));
		std::memset(hostPtrArray, 0, sizeof(VoxelClusterStore*) * arraySize);

		for (std::pair<Vector3i, std::unordered_map<uint32_t, uint32_t>> pair : voxelSceneStorage)
		{
			Vector3i arrayLocation = pair.first - minCoords;
			uint32_t arrayIndex = arrayLocation.getX() + arrayLocation.getY() * arrayDiameter.getX() + arrayLocation.getZ() * arrayDiameter.getX() * arrayDiameter.getY();

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

	Vector3i getArrayDiameter()
	{
		return maxCoords - minCoords + Vector3i(1, 1, 1);
	}

	uint32_t getArraySize()
	{
		Vector3i arrayDiameters = getArrayDiameter();
		return arrayDiameters.getX() * arrayDiameters.getY() * arrayDiameters.getZ();
	}

	Vector3i getMinCoords()
	{
		return minCoords;
	}

	VoxelClusterStore** deviceVoxelScene;

private:
	//Map will be converted to an array of VoxelStructures which contain the actual voxels
	std::unordered_map<Vector3i, std::unordered_map<uint32_t, uint32_t>, Vector3iHashFunction> voxelSceneStorage;
	//Keeps track of the minumum and maximum coordinates to calculate the radius to allocate
	Vector3i minCoords = Vector3i(0, 0, 0);
	Vector3i maxCoords = Vector3i(0, 0, 0);
};
