#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <math.h>

#include "../math/Vector3.cuh"
#include "VoxelSceneInfo.cuh"
#include "camera/Camera.cuh"
#include "rays/Ray.cuh"

#include "../storage/StorageStructure.cuh"

#include "../geometry/VoxelFunctions.cuh"

//Start with 1920x1080 HD image
//Split up the image into 30x30 sections - the GCD is 120 (only 1024 threads allowed per block)

__device__ float applyCeilAndPosEpsilon(float input)
{
	return ceilf(input) + EPSILON;
}

__device__ float applyFloorAndNegEpsilon(float input)
{
	return floorf(input) - EPSILON;
}

__device__ __forceinline__ uint32_t applyDirectionalLightingToColor(uint32_t voxelColor, const Vector3f& normal)
{
	//Find the dot product of the normal and light direction to get the factor
	float diff = max(dot(normal, LIGHT_DIRECTION), 0.0f);
	Vector3f diffuse = diff * LIGHT_COLOR;
	//Convert the color to a vector to calculate its lighting
	Vector3f color = voxelfunc::convertRGBIntegerColorToVector(voxelColor);
	//Convert the color back to its integer representation
	return voxelfunc::convertRGBVectorToInteger(color * diffuse);
}

__device__ __forceinline__ uint32_t applyPointLightingToColor(uint32_t voxelColor, const Vector3f& voxelPosition, const Vector3f& normal)
{
	//Find the Vector distance between the light position and the voxel position
	Vector3f pointToLightDifference = LIGHT_POSITION - voxelPosition;
	//Get the distance between the point and light
	float distance = pointToLightDifference.length();
	//Find the direction to the light source
	Vector3f lightDir = makeUnitVector(pointToLightDifference);
	//Calculate the light attenuation value
	float attenuation = 1 / (LIGHT_CONSTANT + LIGHT_LINEAR * distance + LIGHT_QUADRATIC * (distance * distance));
	//Find the dot product of the normal and light direction to get lighting factor
	float diff = max(dot(normal, lightDir), 0.0f);
	Vector3f diffuse = diff * LIGHT_COLOR;
	//Convert the color to a vector to calculate its lighting
	Vector3f color = voxelfunc::convertRGBIntegerColorToVector(voxelColor);
	Vector3f result = (diffuse * attenuation) * color;
	//Convert the color back to its integer representation
	return voxelfunc::convertRGBVectorToInteger((diffuse * attenuation) * color);
}

__device__ __forceinline__ Vector3f getHitLocation(const Vector3f& voxelStructureTranslationVector, const Vector3f& localRayLocation)
{
	return voxelStructureTranslationVector + localRayLocation;
}

__device__ __forceinline__ bool isRayInRegion(const Ray& ray)
{
	return ray.getOrigin().getX() >= 0.0f && ray.getOrigin().getX() < BLOCK_SIZE &&
		ray.getOrigin().getY() >= 0.0f && ray.getOrigin().getY() < BLOCK_SIZE &&
		ray.getOrigin().getZ() >= 0.0f && ray.getOrigin().getZ() < BLOCK_SIZE;
}

__device__ Ray calculateNextRayPosition(const Ray& currentRay, float (*nextXFunc)(float), float (*nextYFunc)(float), float (*nextZFunc)(float))
{
	//Calculate the next voxel locations

	float nextX = nextXFunc(currentRay.getOrigin().getX());
	float nextY = nextYFunc(currentRay.getOrigin().getY());
	float nextZ = nextZFunc(currentRay.getOrigin().getZ());
	//Calculate the t-values along the ray
	float tX = currentRay.getDirection().getX() != 0.0f ? (nextX - currentRay.getOrigin().getX()) / currentRay.getDirection().getX() : INFINITY;
	float tY = currentRay.getDirection().getY() != 0.0f ? (nextY - currentRay.getOrigin().getY()) / currentRay.getDirection().getY() : INFINITY;
	float tZ = currentRay.getDirection().getZ() != 0.0f ? (nextZ - currentRay.getOrigin().getZ()) / currentRay.getDirection().getZ() : INFINITY;
	//Find the minimum t-value
	float tMin = min(tX, min(tY, tZ));

	//Create the ray at the next position
	return Ray(currentRay.getOrigin() + (tMin + EPSILON) * currentRay.getDirection(), currentRay.getDirection());
}

__device__ Ray calculateNextRayPositionForCluster(const Ray& currentRay, int32_t voxelX, int32_t voxelY, int32_t voxelZ)
{
	//Skip to the next block location
	int32_t nextX = currentRay.getDirection().getX() > 0.0f ? ((voxelX / CLUSTER_SIZE) + 1) * CLUSTER_SIZE : (voxelX / CLUSTER_SIZE) * CLUSTER_SIZE;
	int32_t nextY = currentRay.getDirection().getY() > 0.0f ? ((voxelY / CLUSTER_SIZE) + 1) * CLUSTER_SIZE : (voxelY / CLUSTER_SIZE) * CLUSTER_SIZE;
	int32_t nextZ = currentRay.getDirection().getZ() > 0.0f ? ((voxelZ / CLUSTER_SIZE) + 1) * CLUSTER_SIZE : (voxelZ / CLUSTER_SIZE) * CLUSTER_SIZE;
	//Calculate the t-values along the ray
	float tX = currentRay.getDirection().getX() != 0.0f ? (nextX - currentRay.getOrigin().getX()) / currentRay.getDirection().getX() : INFINITY;
	float tY = currentRay.getDirection().getY() != 0.0f ? (nextY - currentRay.getOrigin().getY()) / currentRay.getDirection().getY() : INFINITY;
	float tZ = currentRay.getDirection().getZ() != 0.0f ? (nextZ - currentRay.getOrigin().getZ()) / currentRay.getDirection().getZ() : INFINITY;
	//Find the minimum t-value
	float tMin = min(tX, min(tY, tZ));

	return  Ray(currentRay.getOrigin() + (tMin + EPSILON) * currentRay.getDirection(), currentRay.getDirection());
}

__device__ Ray calculateNextLocalRayWithRegionDiffs(const Ray& localRay, Vector3i& currentRegion)
{
	//Find the difference of the next region coordinates based on the processed ray from the previous ray marching algorithm
	int32_t regionXDiff = static_cast<int32_t>(localRay.getOrigin().getX() / BLOCK_SIZE + 1) - 1;
	int32_t regionYDiff = static_cast<int32_t>(localRay.getOrigin().getY() / BLOCK_SIZE + 1) - 1;
	int32_t regionZDiff = static_cast<int32_t>(localRay.getOrigin().getZ() / BLOCK_SIZE + 1) - 1;
	//Update the region coordiantes to indicate the current region
	currentRegion[0] += regionXDiff;
	currentRegion[1] += regionYDiff;
	currentRegion[2] += regionZDiff;
	//Update the localRay to be in the space of the new region
	return localRay.convertRayToLocalSpace(Vector3f(regionXDiff * BLOCK_SIZE, regionYDiff * BLOCK_SIZE, regionZDiff * BLOCK_SIZE), 1.0f);
}

__device__ Ray calculateNextRayForRegion(const Ray& localRay, Vector3i& currentRegion)
{
	float nextX = localRay.getDirection().getX() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
	float nextY = localRay.getDirection().getY() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
	float nextZ = localRay.getDirection().getZ() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;

	float tX = (nextX - localRay.getOrigin().getX()) / localRay.getDirection().getX();
	float tY = (nextY - localRay.getOrigin().getY()) / localRay.getDirection().getY();
	float tZ = (nextZ - localRay.getOrigin().getZ()) / localRay.getDirection().getZ();
	//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
	float tMin = min(tX, min(tY, tZ));

	Ray ray = Ray(localRay.getOrigin() + tMin * localRay.getDirection(), localRay.getDirection());
	//Update the localRay to be in the space of the new region
	return calculateNextLocalRayWithRegionDiffs(ray, currentRegion);
}

__device__ float findTValueForClosestRegion(const Ray& sceneRay, const VoxelSceneInfo* sceneInfo)
{
	//Find the closest next location along the ray at the edge of the scene
	int32_t nextX = sceneRay.getDirection().getX() < 0.0f ? sceneInfo->arrDiameter.getX() + sceneInfo->minCoords.getX() : 0 + sceneInfo->minCoords.getX();
	int32_t nextY = sceneRay.getDirection().getY() < 0.0f ? sceneInfo->arrDiameter.getY() + sceneInfo->minCoords.getY() : 0 + sceneInfo->minCoords.getY();
	int32_t nextZ = sceneRay.getDirection().getZ() < 0.0f ? sceneInfo->arrDiameter.getZ() + sceneInfo->minCoords.getZ() : 0 + sceneInfo->minCoords.getZ();
	//Find the ray parameter values
	float tX = (nextX * BLOCK_SIZE - sceneRay.getOrigin().getX()) / sceneRay.getDirection().getX();
	float tY = (nextY * BLOCK_SIZE - sceneRay.getOrigin().getY()) / sceneRay.getDirection().getY();
	float tZ = (nextZ * BLOCK_SIZE - sceneRay.getOrigin().getZ()) / sceneRay.getDirection().getZ();
	//Ensure that negative values are excluded from the min calculation because negative values mean they are behind the ray
	if (tX <= 0.0f) tX = INFINITY;
	if (tY <= 0.0f) tY = INFINITY;
	if (tZ <= 0.0f) tZ = INFINITY;

	return min(tX, min(tY, tZ));
}

__device__ uint32_t shadowRayMarchVoxelGrid(Ray& ray, const Vector3f regionWorldPosition, VoxelClusterStore* storageStructure)
{
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;

	//Create the ray at the next position
	ray = calculateNextRayPosition(ray, nextXFunc, nextYFunc, nextZFunc);

	while (isRayInRegion(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t voxelX = static_cast<int32_t>(ray.getOrigin().getX());
		int32_t voxelY = static_cast<int32_t>(ray.getOrigin().getY());
		int32_t voxelZ = static_cast<int32_t>(ray.getOrigin().getZ());

		//Checks if a "voxel space" exists and moves the ray until it is in a valid voxel space (i.e. if a Voxel Cluster does not exist, skip to the next cluster)
		if (!storageStructure->doesClusterExist(voxelX, voxelY, voxelZ))
		{
			//Create the ray at the next position
			ray = calculateNextRayPositionForCluster(ray, voxelX, voxelY, voxelZ);
			continue;
		}
		//Check if a voxel exists at the given voxel space
		uint32_t voxelColor = storageStructure->lookupVoxel(voxelX, voxelY, voxelZ);
		if (voxelColor != EMPTY_KEY)
		{
			return voxelColor;
		}

		//Create the ray at the next position
		ray = calculateNextRayPosition(ray, nextXFunc, nextYFunc, nextZFunc);
	}

	//Return EMPTY_VAL to indicate no intersection was made in the region
	return EMPTY_VAL;
}

__device__ bool isInShadowOriginalRayMarch(Ray localRay, const VoxelSceneInfo* sceneInfo, Vector3i currentRegion)
{
	if (!USE_SHADOWS)
	{
		return false;
	}

	//while the region coordinates are within the bounds of the scene, keep doing calculation
	while (sceneInfo->isRegionInScene(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ()))
	{
		VoxelClusterStore* regionStorageStructure = sceneInfo->getRegionStorageStructure(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ());
		while (regionStorageStructure == nullptr)
		{
			//Update the localRay to be in the space of the new region
			localRay = calculateNextRayForRegion(localRay, currentRegion);
			if (sceneInfo->isRegionInScene(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ()))
				regionStorageStructure = sceneInfo->getRegionStorageStructure(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ());
			else return false;
		}

		//Vector is passed along for lighting calculations
		Vector3f regionWorldPosition = sceneInfo->translationVector + Vector3f( (currentRegion.getX() - 1) * BLOCK_SIZE, (currentRegion.getY() - 1) * BLOCK_SIZE, (currentRegion.getZ() - 1) * BLOCK_SIZE);
		//call the raymarching function for that voxel structure
		uint32_t voxelColor = shadowRayMarchVoxelGrid(localRay, regionWorldPosition, regionStorageStructure);
		if (voxelColor != EMPTY_VAL)
		{
			return true;
		}
		//Update the localRay to be in the space of the new region
		localRay = calculateNextLocalRayWithRegionDiffs(localRay, currentRegion);
	}

	//Return the background color
	return false;
}

__device__ __forceinline__ Vector3f getNormalFromRay(const Ray& ray)
{
	Vector3f normal;
	//Find which ray component is esentially equal to its nearest integer. That will decide which is the normal
	if (std::abs(ray.getOrigin().getX() - std::floorf(ray.getOrigin().getX())) < EPSILON)
		normal = Vector3f(copysignf(1.0f, -ray.getDirection().getX()), 0.0f, 0.0f);
	else if (std::abs(ray.getOrigin().getY() - std::floorf(ray.getOrigin().getY())) < EPSILON)
		normal = Vector3f(0.0f, copysignf(1.0f, -ray.getDirection().getY()), 0.0f);
	else
		normal = Vector3f(0.0f, 0.0f, copysignf(1.0f, -ray.getDirection().getZ()));
	return normal;
}

__device__ __forceinline__ uint32_t applyLighting(uint32_t voxelColor, const Vector3f& normal, const Vector3f& regionWorldPosition, const Vector3f& rayOrigin)
{
	if (USE_POINT_LIGHT)
	{
		Vector3f hitPoint = getHitLocation(regionWorldPosition, rayOrigin);
		return applyPointLightingToColor(voxelColor, hitPoint, normal);
	}
	
	return applyDirectionalLightingToColor(voxelColor, normal);
}

__device__ uint32_t rayMarchVoxelRegion(Ray& ray, const Vector3f regionWorldPosition, VoxelClusterStore* storageStructure, const VoxelSceneInfo* voxelSceneInfo, Vector3i currentRegion)
{
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;

	//Create the ray at the next position
	ray = calculateNextRayPosition(ray, nextXFunc, nextYFunc, nextZFunc);

	while (isRayInRegion(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t voxelX = static_cast<int32_t>(ray.getOrigin().getX());
		int32_t voxelY = static_cast<int32_t>(ray.getOrigin().getY());
		int32_t voxelZ = static_cast<int32_t>(ray.getOrigin().getZ());

		//Checks if a "voxel space" exists and moves the ray until it is in a valid voxel space (i.e. if a Voxel Cluster does not exist, skip to the next cluster)
		if (!storageStructure->doesClusterExist(voxelX, voxelY, voxelZ))
		{
			//Create the ray at the next position
			ray = calculateNextRayPositionForCluster(ray, voxelX, voxelY, voxelZ);
			continue;
		}
		//Check if a voxel exists at the given voxel space
		uint32_t voxelColor = storageStructure->lookupVoxel(voxelX, voxelY, voxelZ);
		if (voxelColor != EMPTY_KEY)
		{
			//Find the normal based on which axis is hit
			Vector3f normal = getNormalFromRay(ray);
			//Apply lighting to the color value
			uint32_t resultingColor = applyLighting(voxelColor, normal, regionWorldPosition, ray.getOrigin());
			return resultingColor * !isInShadowOriginalRayMarch(Ray(ray.getOrigin(), LIGHT_DIRECTION), voxelSceneInfo, currentRegion);
		}

		//Create the ray at the next position
		ray = calculateNextRayPosition(ray, nextXFunc, nextYFunc, nextZFunc);
	}

	//Return EMPTY_VAL to indicate no intersection was made in the region
	return EMPTY_VAL;
}

__device__ uint32_t rayMarchVoxelScene(const Ray& originalRay, const VoxelSceneInfo* sceneInfo)
{
	//Convert the world ray into the local coordinate system of the scene
	Ray sceneRay = originalRay.convertRayToLocalSpace(sceneInfo->translationVector, sceneInfo->scale);

	//determine the region coordinates of the ray
	Vector3i currentRegion = Vector3i(static_cast<int32_t>(sceneRay.getOrigin().getX() / BLOCK_SIZE + 1),
		static_cast<int32_t>(sceneRay.getOrigin().getY() / BLOCK_SIZE + 1),
		static_cast<int32_t>(sceneRay.getOrigin().getZ() / BLOCK_SIZE + 1));

	//Try to place the ray in the scene's boundaries if it outside, if that is not possible quit ray marching
	while (!sceneInfo->isRegionInScene(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ()))
	{
		float tMin = findTValueForClosestRegion(sceneRay, sceneInfo);
		if (tMin == INFINITY) return 0;

		sceneRay = Ray(sceneRay.getOrigin() + (tMin + EPSILON) * sceneRay.getDirection(), sceneRay.getDirection());

		currentRegion = Vector3i(static_cast<int32_t>(sceneRay.getOrigin().getX() / BLOCK_SIZE + 1),
			static_cast<int32_t>(sceneRay.getOrigin().getY() / BLOCK_SIZE + 1),
			static_cast<int32_t>(sceneRay.getOrigin().getZ() / BLOCK_SIZE + 1));
	}

	//Transform the ray into the local coordinates of the current region it is located in
	Ray localRay = sceneRay.convertRayToLocalSpace(
		Vector3f( (currentRegion.getX() - 1) * BLOCK_SIZE, (currentRegion.getY() - 1) * BLOCK_SIZE, (currentRegion.getZ() - 1) * BLOCK_SIZE), 1.0f
	);

	//while the region coordinates are within the bounds of the scene, keep doing calculation
	while (sceneInfo->isRegionInScene(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ()))
	{
		VoxelClusterStore* regionStorageStructure = sceneInfo->getRegionStorageStructure(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ());
		while (regionStorageStructure == nullptr)
		{
			//Update the localRay to be in the space of the new region
			localRay = calculateNextRayForRegion(localRay, currentRegion);
			if (sceneInfo->isRegionInScene(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ()))
				regionStorageStructure = sceneInfo->getRegionStorageStructure(currentRegion.getX(), currentRegion.getY(), currentRegion.getZ());
			else return 0;
		}

		//Vector is passed along for lighting calculations
		Vector3f regionWorldPosition = sceneInfo->translationVector + Vector3f( (currentRegion.getX() - 1) * BLOCK_SIZE, (currentRegion.getY() - 1) * BLOCK_SIZE, (currentRegion.getZ() - 1) * BLOCK_SIZE);
		//call the raymarching function for that voxel structure
		uint32_t voxelColor = rayMarchVoxelRegion(localRay, regionWorldPosition, regionStorageStructure, sceneInfo, currentRegion);
		if (voxelColor != EMPTY_VAL)
		{
			return voxelColor;
		}
		//Update the localRay to be in the space of the new region
		localRay = calculateNextLocalRayWithRegionDiffs(localRay, currentRegion);
	}

	//Return the background color
	return 0;
}

__forceinline__ __device__ Ray calculateWorldRay(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t imgHeight, const Camera* camera)
{
	//Calculate normalized camera space coordinates (between 0 and 1) of the center of the pixel
	float uCamera = (xPixel + 0.5f) / static_cast<float>(imgWidth);
	//Since framebuffer coordinates have posative y going down and 3D position has a positive y going up. The imgHeight needs to be subracted to make up for this difference
	float vCamera = (imgHeight - yPixel + 0.5f) / static_cast<float>(imgHeight);

	//Calculate the ray cast by the camera with the camera space coordiantes for the pixel
	return camera->generateRay(uCamera, vCamera);
}

__forceinline__ __device__ void writeColorToFramebuffer(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t color, uint8_t* framebuffer)
{
	//Set the framebuffer location for that pixel to the returned color
	uint32_t pixelIndex = yPixel * imgWidth * 3 + xPixel * 3;
	framebuffer[pixelIndex] = voxelfunc::getRedComponent(color);
	framebuffer[pixelIndex + 1] = voxelfunc::getGreenComponent(color);
	framebuffer[pixelIndex + 2] = voxelfunc::getBlueComponent(color);
}

__global__ void rayMarchSceneOriginal(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelSceneInfo* voxelSceneInfo, uint8_t* framebuffer)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray worldRay = calculateWorldRay(xPixel, yPixel, imgWidth, imgHeight, camera);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelScene(worldRay, voxelSceneInfo);

	writeColorToFramebuffer(xPixel, yPixel, imgWidth, color, framebuffer);
}