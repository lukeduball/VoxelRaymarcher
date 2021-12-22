#pragma once

#include "rays/Ray.cuh"
#include "../storage/StorageStructure.cuh"

class VoxelSceneInfo
{
public:
	__host__ __device__ VoxelSceneInfo() : translationVector(Vector3f()), scale(1) {}

	__host__ __device__ VoxelSceneInfo(VoxelClusterStore** scene, uint32_t arrDia, int32_t mCoord, Vector3f location, uint32_t scale = 1) : 
		sceneStorage(scene), arrDiameter(arrDia), minCoord(mCoord), translationVector(location), scale(scale) {}

	__device__ __forceinline__ VoxelClusterStore* getRegionStorageStructure(int32_t x, int32_t y, int32_t z) const
	{
		uint32_t uX = x - minCoord;
		uint32_t uY = y - minCoord;
		uint32_t uZ = z - minCoord;
		ASSERT(uX < arrDiameter && uY < arrDiameter && uZ < arrDiameter, "");
		return sceneStorage[uX + uY * arrDiameter + uZ * arrDiameter * arrDiameter];
	}

	__device__ __forceinline__ bool isRayInScene(int32_t x, int32_t y, int32_t z) const
	{
		uint32_t uX = x - minCoord;
		uint32_t uY = y - minCoord;
		uint32_t uZ = z - minCoord;
		return uX < arrDiameter && uY < arrDiameter && uZ < arrDiameter;
	}

	__device__ __forceinline__ bool isRegionInScene(int32_t regionX, int32_t regionY, int32_t regionZ)
	{
		int32_t x = regionX - minCoord;
		int32_t y = regionY - minCoord;
		int32_t z = regionZ - minCoord;
		return x > 0 && y > 0 && z > 0 && x < arrDiameter && y < arrDiameter && z < arrDiameter;
	}

	Vector3f translationVector;
	uint32_t scale;

	VoxelClusterStore** sceneStorage;
	uint32_t arrDiameter;
	int32_t minCoord;
};