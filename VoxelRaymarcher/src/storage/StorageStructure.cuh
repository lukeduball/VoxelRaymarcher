#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "../renderer/rays/Ray.cuh"

#include "CuckooHashTable.cuh"
#include "VoxelClusterStore.cuh"

class StorageStructure
{
public:
	__device__ virtual uint32_t lookupVoxel(int32_t x, int32_t y, int32_t z) const = 0;
	__device__ virtual bool doesVoxelSpaceExist(int32_t x, int32_t y, int32_t z) const = 0;
};

class VCSStorageStructure : public StorageStructure
{
public:
	__device__ VCSStorageStructure(VoxelClusterStore* voxelClusterStore) : voxelClusterStorePtr(voxelClusterStore) {}

	__device__ virtual uint32_t lookupVoxel(int32_t x, int32_t y, int32_t z) const override
	{
		return voxelClusterStorePtr->lookupVoxel(x, y, z);
	}

	__device__ virtual bool doesVoxelSpaceExist(int32_t x, int32_t y, int32_t z) const override
	{
		return voxelClusterStorePtr->doesClusterExist(x, y, z);
	}

private:
	VoxelClusterStore* voxelClusterStorePtr;
};

class HashTableStorageStructure : public StorageStructure
{
public:
	__device__ HashTableStorageStructure(CuckooHashTable* hashTable) : hashTablePtr(hashTable) {}

	__device__ virtual uint32_t lookupVoxel(int32_t x, int32_t y, int32_t z) const override
	{
		return hashTablePtr->lookupVoxel(x, y, z);
	}

	//The hash table does not split voxel into spaces therefore the voxel space always exists
	__device__ virtual bool doesVoxelSpaceExist(int32_t x, int32_t y, int32_t z) const override
	{
		return true;
	}

private:
	CuckooHashTable* hashTablePtr;
};