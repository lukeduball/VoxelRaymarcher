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
		uint32_t uX = x - 1 - minCoord;
		uint32_t uY = y - 1 - minCoord;
		uint32_t uZ = z - 1 - minCoord;
		ASSERT(uX < arrDiameter && uY < arrDiameter && uZ < arrDiameter, "");
		return sceneStorage[uX + uY * arrDiameter + uZ * arrDiameter * arrDiameter];
	}

	__device__ __forceinline__ bool isRegionInScene(int32_t regionX, int32_t regionY, int32_t regionZ) const
	{
		uint32_t uX = regionX - 1 - minCoord;
		uint32_t uY = regionY - 1 - minCoord;
		uint32_t uZ = regionZ - 1 - minCoord;
		return uX < arrDiameter && uY < arrDiameter && uZ < arrDiameter;
	}


	Vector3f translationVector;
	uint32_t scale;

	VoxelClusterStore** sceneStorage;
	uint32_t arrDiameter;
	int32_t minCoord;
};