#pragma once

#include "../../math/Vector3.cuh"

class Ray
{
public:
	__host__ __device__ Ray(const Vector3f& o, const Vector3f& dir) : origin(o), direction(dir) {}

	__host__ __device__ inline Vector3f getOrigin() const { return origin; }
	__host__ __device__ inline Vector3f getDirection() const { return direction; }

	__host__ __device__ Ray convertRayToLocalSpace(Vector3f translation, uint32_t scale) const
	{
		return Ray(origin - (translation * static_cast<float>(scale)), direction);
	}

	__host__ __device__ Ray convertRayToLongestAxisDirection(const Ray& ray, uint32_t& longestAxis, uint32_t& shortAxis1, uint32_t& shortAxis2) const
	{
		float xDirAbs = fabsf(ray.getDirection().getX());
		float yDirAbs = fabsf(ray.getDirection().getY());
		float zDirAbs = fabsf(ray.getDirection().getZ());
		float constant = 0.0f;
		if (xDirAbs > yDirAbs && xDirAbs > zDirAbs)
		{
			longestAxis = 0;
			if (yDirAbs > zDirAbs)
			{
				shortAxis1 = 1;
				shortAxis2 = 2;
			}
			else
			{
				shortAxis1 = 2;
				shortAxis2 = 1;
			}
			constant = 1.0f / xDirAbs;
		}
		else if (yDirAbs > zDirAbs)
		{
			longestAxis = 1;
			if (xDirAbs > zDirAbs)
			{
				shortAxis1 = 0;
				shortAxis2 = 2;
			}
			else
			{
				shortAxis1 = 2;
				shortAxis2 = 0;
			}
			constant = 1.0f / yDirAbs;
		}
		else
		{
			longestAxis = 2;
			if (xDirAbs > yDirAbs)
			{
				shortAxis1 = 0;
				shortAxis2 = 1;
			}
			else
			{
				shortAxis1 = 1;
				shortAxis2 = 0;
			}
			constant = 1.0f / zDirAbs;
		}
		return Ray(ray.getOrigin(), ray.getDirection() * constant);
	}


private:
	Vector3f origin;
	Vector3f direction;
};