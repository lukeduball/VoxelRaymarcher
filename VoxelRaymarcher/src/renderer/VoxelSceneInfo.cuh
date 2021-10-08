#pragma once

#include "rays/Ray.cuh"

class VoxelSceneInfo
{
public:
	__host__ __device__ VoxelSceneInfo() : translationVector(Vector3f()), scale(1) {}

	__host__ __device__ VoxelSceneInfo(Vector3f location, uint32_t scale = 1) : 
		translationVector(location), scale(scale) {}

	Vector3f translationVector;
	uint32_t scale;
};