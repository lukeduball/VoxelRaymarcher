#pragma once

#include <unordered_map>
#include <vector>

#include "VoxelFunctions.cuh"

#include "../storage/VoxelClusterStore.cuh"
#include "../storage/CuckooHashTable.cuh"

class VoxelSceneCPU
{
public:
	void insertVoxel(int32_t x, int32_t y, int32_t z, uint32_t color)
	{
		int32_t voxelStructureIDX = x / BLOCK_SIZE;
		int32_t voxelStructureIDY = y / BLOCK_SIZE;
		int32_t voxelStructureIDZ = z / BLOCK_SIZE;
		
		//To account for negative values, add the BLOCK_SIZE to the remainder and find the remainer again
		uint32_t localVoxelX = ((x % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;
		uint32_t localVoxelY = ((y % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;
		uint32_t localVoxelZ = ((z % BLOCK_SIZE) + BLOCK_SIZE) % BLOCK_SIZE;

		int32_t minVoxelRegionCoord = std::min(voxelStructureIDX, std::min(voxelStructureIDY, voxelStructureIDZ));
		int32_t maxVoxelRegionCoord = std::max(voxelStructureIDX, std::max(voxelStructureIDY, voxelStructureIDZ));

		if (minVoxelRegionCoord < minCoord)
			minCoord = minVoxelRegionCoord;
		if (maxVoxelRegionCoord > maxCoord)
			maxCoord = maxVoxelRegionCoord;


		Vector3i storageKey = Vector3i(voxelStructureIDX, voxelStructureIDY, voxelStructureIDZ);
		if (voxelSceneStorage.find(storageKey) == voxelSceneStorage.end())
		{
			voxelSceneStorage[storageKey] = std::unordered_map<uint32_t, uint32_t>();
		}
		voxelSceneStorage[storageKey][voxelfunc::generate3DPoint(localVoxelX, localVoxelY, localVoxelZ)] = color;
	}

	void generateVoxelScene(StorageType storageType)
	{
		int32_t arrayDiameter = maxCoord - minCoord + 1;
		uint32_t arraySize = arrayDiameter * arrayDiameter * arrayDiameter;

		void** hostPtrArray = static_cast<void**>(malloc(sizeof(void*) * arraySize));
		std::memset(hostPtrArray, 0, sizeof(void*) * arraySize);

		for (std::pair<Vector3i, std::unordered_map<uint32_t, uint32_t>> pair : voxelSceneStorage)
		{
			Vector3i arrayLocation = pair.first - Vector3i(minCoord, minCoord, minCoord);
			uint32_t arrayIndex = arrayLocation.getX() + arrayLocation.getY() * arrayDiameter + arrayLocation.getZ() * arrayDiameter * arrayDiameter;

			switch (storageType)
			{
			case StorageType::VOXEL_CLUSTER_STORE: {
				//Generate the voxel cluster store on the CPU
				VoxelClusterStore* voxelClusterStore = new VoxelClusterStore(pair.second);
				VoxelClusterStore* devicePtr;
				//Move the voxel cluster store to the GPU
				cudaMalloc(&devicePtr, sizeof(VoxelClusterStore));
				cudaMemcpy(devicePtr, voxelClusterStore, sizeof(VoxelClusterStore), cudaMemcpyHostToDevice);
				hostPtrArray[arrayIndex] = devicePtr;
				break; }
			case StorageType::HASH_TABLE: {
				//Generate the voxel cluster store on the CPU
				CuckooHashTable* hashTable = new CuckooHashTable(pair.second);
				CuckooHashTable* devicePtr;
				//Move the voxel cluster store to the GPU
				cudaMalloc(&devicePtr, sizeof(CuckooHashTable));
				cudaMemcpy(devicePtr, hashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);
				hostPtrArray[arrayIndex] = devicePtr;
				break; }
			}
		}

		cudaMalloc(&deviceVoxelScene, sizeof(void*) * arraySize);
		cudaMemcpy(deviceVoxelScene, hostPtrArray, sizeof(void*) * arraySize, cudaMemcpyHostToDevice);

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
		cudaFree(deviceVoxelScene);
	}

	uint32_t getArraySize()
	{
		int32_t arrayDiameter = maxCoord - minCoord + 1;
		return arrayDiameter * arrayDiameter * arrayDiameter;
	}

	void** deviceVoxelScene;

private:
	//Map will be converted to an array of VoxelStructures which contain the actual voxels
	std::unordered_map<Vector3i, std::unordered_map<uint32_t, uint32_t>, Vector3iHashFunction> voxelSceneStorage;
	//Keeps track of the minumum and maximum coordinates to calculate the radius to allocate
	int32_t minCoord = 0;
	int32_t maxCoord = 0;
};
