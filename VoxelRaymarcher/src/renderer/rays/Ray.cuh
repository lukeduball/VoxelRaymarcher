#pragma once

#include "../../math/Vector3.cuh"

class Ray
{
public:
	__host__ __device__ Ray(const Vector3& o, const Vector3& dir) : origin(o), direction(dir) {}

	__host__ __device__ inline Vector3 getOrigin() const { return origin; }
	__host__ __device__ inline Vector3 getDirection() const { return direction; }

	__host__ __device__ Ray convertRayToLocalSpace(Vector3 translation, uint32_t scale)
	{
		return Ray(origin - (translation * static_cast<float>(scale)), direction);
	}


private:
	Vector3 origin;
	Vector3 direction;
};