#pragma once

#include "../cuckoohash/CuckooHashTable.cuh"
#include "rays/Ray.cuh"

class VoxelStructure
{
public:
	VoxelStructure(CuckooHashTable* table, Vector3 location, uint32_t s, uint32_t scale = 1) : 
		hashTable(table), translationVector(location), size(s), scale(scale) {}

	__device__ bool isRayInStructure(const Ray& ray) const
	{
		return ray.getOrigin().getX() >= 0.0f && ray.getOrigin().getX() < size &&
			ray.getOrigin().getY() >= 0.0f && ray.getOrigin().getY() < size &&
			ray.getOrigin().getZ() >= 0.0f && ray.getOrigin().getZ() < size;
	}

	Vector3 translationVector;
	uint32_t size;
	uint32_t scale;

	CuckooHashTable* hashTable;
};