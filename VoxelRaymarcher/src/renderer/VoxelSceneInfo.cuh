#pragma once

#include "rays/Ray.cuh"
#include "../storage/StorageStructure.cuh"

class VoxelSceneInfo
{
public:
	__host__ __device__ VoxelSceneInfo() : translationVector(Vector3f()), scale(1) {}

	__host__ __device__ VoxelSceneInfo(VoxelClusterStore** scene, Vector3i arrDia, Vector3i mCoord, Vector3f location, uint32_t scale = 1) : 
		sceneStorage(scene), arrDiameter(arrDia), minCoords(mCoord), translationVector(location), scale(scale) {}

	__device__ __forceinline__ VoxelClusterStore* getRegionStorageStructure(int32_t x, int32_t y, int32_t z) const
	{
		uint32_t uX = x - 1 - minCoords.getX();
		uint32_t uY = y - 1 - minCoords.getY();
		uint32_t uZ = z - 1 - minCoords.getZ();
		ASSERT(uX < arrDiameter.getX() && uY < arrDiameter.getY() && uZ < arrDiameter.getZ(), "");
		return sceneStorage[uX + uY * arrDiameter.getX() + uZ * arrDiameter.getX() * arrDiameter.getY()];
	}

	__device__ __forceinline__ bool isRegionInScene(int32_t regionX, int32_t regionY, int32_t regionZ) const
	{
		uint32_t uX = regionX - 1 - minCoords.getX();
		uint32_t uY = regionY - 1 - minCoords.getY();
		uint32_t uZ = regionZ - 1 - minCoords.getZ();
		return uX < arrDiameter.getX() && uY < arrDiameter.getY() && uZ < arrDiameter.getZ();
	}


	Vector3f translationVector;
	uint32_t scale;

	VoxelClusterStore** sceneStorage;
	Vector3i arrDiameter;
	Vector3i minCoords;
};