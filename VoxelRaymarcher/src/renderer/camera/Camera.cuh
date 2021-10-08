#pragma once

#include "../../math/MathConstants.cuh"
#include "../../math/Vector3.cuh"

#include "../rays/Ray.cuh"

class Camera
{
public:
	__host__ __device__ Camera(Vector3f o, Vector3f lookAt, Vector3f globalUp, float fieldOfView, float aspectRatio)
	{
		float halfHeight = tanf((fieldOfView * PI / 180.f) / 2.0f);
		float halfWidth = halfHeight * aspectRatio;
		origin = o;
		Vector3f w = makeUnitVector(lookAt - origin);
		Vector3f u = makeUnitVector(cross(w, globalUp));
		Vector3f v = cross(u, w);
		lowerLeftCorner = origin - halfWidth * u - halfHeight * v + w;
		forwardVector = w;
		horizontalVector = 2 * halfWidth * u;
		verticalVector = 2 * halfHeight * v;
	}

	__host__ __device__ Ray generateRay(float u, float v) const
	{
		Vector3f rayOrigin = lowerLeftCorner + u * horizontalVector + v * verticalVector;
		return Ray(rayOrigin, makeUnitVector(rayOrigin - origin)); 
	}

	Vector3f origin;
	Vector3f lowerLeftCorner;
	//Vectors are in camera space
	Vector3f horizontalVector;
	Vector3f verticalVector;
	Vector3f forwardVector;
};