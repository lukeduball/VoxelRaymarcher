#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "../renderer/rays/Ray.cuh"

#include "CuckooHashTable.cuh"
#include "VoxelClusterStore.cuh"

class StorageStructure
{
public:
	__device__ virtual uint32_t lookupVoxel(int32_t* gridValues, Ray& ray) const = 0;
};

class VCSStorageStructure : public StorageStructure
{
public:
	__device__ VCSStorageStructure(VoxelClusterStore* voxelClusterStore) : voxelClusterStorePtr(voxelClusterStore) {}

	__device__ virtual uint32_t lookupVoxel(int32_t* gridValues, Ray& ray) const override
	{
		return voxelClusterStorePtr->lookupVoxel(gridValues, ray);
	}

private:
	VoxelClusterStore* voxelClusterStorePtr;
};

class HashTableStorageStructure : public StorageStructure
{
public:
	__device__ HashTableStorageStructure(CuckooHashTable* hashTable) : hashTablePtr(hashTable) {}

	__device__ virtual uint32_t lookupVoxel(int32_t* gridValues, Ray& ray) const override
	{
		return hashTablePtr->lookupVoxel(gridValues, ray);
	}

private:
	CuckooHashTable* hashTablePtr;
};