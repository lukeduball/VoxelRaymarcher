#pragma once

#include "../../math/Vector3.cuh"
#include "../rays/Ray.cuh"

class Camera
{
public:
	Camera(Vector3 o, Vector3 lookAt, Vector3 globalUp, float fieldOfView, float aspectRatio)
	{
		float halfHeight = tanf((fieldOfView * M_PI / 180.f) / 2.0f);
		float halfWidth = halfHeight * aspectRatio;
		origin = o;
		Vector3 w = makeUnitVector(lookAt - origin);
		Vector3 u = makeUnitVector(cross(w, globalUp));
		Vector3 v = cross(u, w);
		lowerLeftCorner = origin - halfWidth * u - halfHeight * v + w;
		forwardVector = w;
		horizontalVector = 2 * halfWidth * u;
		verticalVector = 2 * halfHeight * v;
	}

	__host__ __device__ Ray generateRay(float u, float v) 
	{
		Vector3 rayOrigin = lowerLeftCorner + u * horizontalVector + v * verticalVector;
		return Ray(rayOrigin, makeUnitVector(rayOrigin - origin)); 
	}

	Vector3 origin;
	Vector3 lowerLeftCorner;
	//Vectors are in camera space
	Vector3 horizontalVector;
	Vector3 verticalVector;
	Vector3 forwardVector;
};