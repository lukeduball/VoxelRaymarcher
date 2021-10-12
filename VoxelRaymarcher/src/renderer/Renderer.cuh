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

__device__ float applyCeilAndPosEpsilon1(float input)
{
	return ceilf(input) + EPSILON;
}

__device__ float applyFloorAndNegEpsilon1(float input)
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
		ray.getOrigin().getZ() >= 0.0f && ray.getOrigin().getY() < BLOCK_SIZE;
}

__device__ bool isInShadowOriginalRayMarch(const Ray& originalRay, StorageStructure* storageStructure)
{
	if (!USE_SHADOWS)
	{
		return false;
	}

	Ray ray = originalRay;
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;

	//Calculate the next voxel location

	float nextX = nextXFunc(ray.getOrigin().getX());
	float nextY = nextYFunc(ray.getOrigin().getY());
	float nextZ = nextZFunc(ray.getOrigin().getZ());
	//Calculate the t-values along the ray
	float tX = ray.getDirection().getX() != 0.0f ? (nextX - ray.getOrigin().getX()) / ray.getDirection().getX() : INFINITY;
	float tY = ray.getDirection().getY() != 0.0f ? (nextY - ray.getOrigin().getY()) / ray.getDirection().getY() : INFINITY;
	float tZ = ray.getDirection().getZ() != 0.0f ? (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ() : INFINITY;
	//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
	float tMin = min(tX, min(tY, tZ));

	//Create the ray at the next position
	ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());

	while (isRayInRegion(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t voxelX = static_cast<int32_t>(ray.getOrigin().getX());
		int32_t voxelY = static_cast<int32_t>(ray.getOrigin().getY());
		int32_t voxelZ = static_cast<int32_t>(ray.getOrigin().getZ());

		//Checks if a "voxel space" exists and moves the ray until it is in a valid voxel space (i.e. if a Voxel Cluster does not exist, skip to the next cluster)
		if (!storageStructure->doesVoxelSpaceExist(voxelX, voxelY, voxelZ))
		{
			//Skip to the next block location
			int32_t nextX = ray.getDirection().getX() > 0.0f ? ((voxelX / 8) + 1) * 8 : (voxelX / 8) * 8 - 1;
			int32_t nextY = ray.getDirection().getY() > 0.0f ? ((voxelY / 8) + 1) * 8 : (voxelY / 8) * 8 - 1;
			int32_t nextZ = ray.getDirection().getZ() > 0.0f ? ((voxelZ / 8) + 1) * 8 : (voxelZ / 8) * 8 - 1;
			//Calculate the t-values along the ray
			float tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
			float tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
			float tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
			//Find the minimum t-value
			float tMin = min(tX, min(tY, tZ));

			//Create the ray at the next position
			ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());
			continue;
		}
		//Check if a voxel exists at the given voxel space
		uint32_t voxelColor = storageStructure->lookupVoxel(voxelX, voxelY, voxelZ);
		if (voxelColor != EMPTY_KEY && voxelColor != FINISH_VAL)
		{
			return true;
		}

		//Calculate the next voxel location

		nextX = nextXFunc(ray.getOrigin().getX());
		nextY = nextYFunc(ray.getOrigin().getY());
		nextZ = nextZFunc(ray.getOrigin().getZ());
		//Calculate the t-values along the ray
		tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
		tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
		tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
		//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
		tMin = min(tX, min(tY, tZ));

		//Create the ray at the next position
		ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());
	}

	//Return the background color of black
	return false;
}

__device__ __forceinline__ bool isRayInScene(uint32_t x, uint32_t y, uint32_t z)
{
	return x < BLOCK_SIZE && y < BLOCK_SIZE && z < BLOCK_SIZE;
}

__device__ uint32_t rayMarchVoxelGrid(Ray& ray, const Vector3f regionWorldPosition, StorageStructure* storageStructure)
{
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;

	//Calculate the next voxel location

	float nextX = nextXFunc(ray.getOrigin().getX());
	float nextY = nextYFunc(ray.getOrigin().getY());
	float nextZ = nextZFunc(ray.getOrigin().getZ());
	//Calculate the t-values along the ray
	float tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
	float tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
	float tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
	//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
	float tMin = min(tX, min(tY, tZ));

	//Create the ray at the next position
	ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());

	while (isRayInRegion(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t voxelX = static_cast<int32_t>(ray.getOrigin().getX());
		int32_t voxelY = static_cast<int32_t>(ray.getOrigin().getY());
		int32_t voxelZ = static_cast<int32_t>(ray.getOrigin().getZ());

		//Checks if a "voxel space" exists and moves the ray until it is in a valid voxel space (i.e. if a Voxel Cluster does not exist, skip to the next cluster)
		if (!storageStructure->doesVoxelSpaceExist(voxelX, voxelY, voxelZ))
		{
			//Skip to the next block location
			int32_t nextX = ray.getDirection().getX() > 0.0f ? ((voxelX / 8) + 1) * 8 : (voxelX / 8) * 8 - 1;
			int32_t nextY = ray.getDirection().getY() > 0.0f ? ((voxelY / 8) + 1) * 8 : (voxelY / 8) * 8 - 1;
			int32_t nextZ = ray.getDirection().getZ() > 0.0f ? ((voxelZ / 8) + 1) * 8 : (voxelZ / 8) * 8 - 1;
			//Calculate the t-values along the ray
			float tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
			float tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
			float tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
			//Find the minimum t-value
			float tMin = min(tX, min(tY, tZ));

			//Create the ray at the next position
			ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());
			continue;
		}
		//Check if a voxel exists at the given voxel space
		uint32_t voxelColor = storageStructure->lookupVoxel(voxelX, voxelY, voxelZ);
		if (voxelColor != EMPTY_KEY && voxelColor != FINISH_VAL)
		{
			//Find the normal based on which axis is hit
			Vector3f normal;
			if (tMin == tX)
			{
				normal = Vector3f(copysignf(1.0f, -ray.getDirection().getX()), 0.0f, 0.0f);
			}
			else if (tMin == tY)
			{
				normal = Vector3f(0.0f, copysignf(1.0f, -ray.getDirection().getY()), 0.0f);
			}
			else
			{
				normal = Vector3f(0.0f, 0.0f, copysignf(1.0f, -ray.getDirection().getZ()));
			}

			uint32_t resultingColor = 0;
			if (USE_POINT_LIGHT)
			{
				Vector3f hitPoint = getHitLocation(regionWorldPosition, ray.getOrigin());
				resultingColor = applyPointLightingToColor(voxelColor, hitPoint, normal);
			}
			else
			{
				resultingColor = applyDirectionalLightingToColor(voxelColor, normal);
			}
			return resultingColor * !isInShadowOriginalRayMarch(Ray(ray.getOrigin(), LIGHT_DIRECTION), storageStructure);
		}

		//Calculate the next voxel location

		nextX = nextXFunc(ray.getOrigin().getX());
		nextY = nextYFunc(ray.getOrigin().getY());
		nextZ = nextZFunc(ray.getOrigin().getZ());
		//Calculate the t-values along the ray
		tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
		tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
		tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
		//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
		tMin = min(tX, min(tY, tZ));

		//Create the ray at the next position
		ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());
	}

	//Return EMPTY_VAL to indicate no intersection was made in the region
	return EMPTY_VAL;
}

__device__ uint32_t rayMarchVoxelScene(const Ray& originalRay, const VoxelSceneInfo* sceneInfo, StorageStructure** voxelScene, uint32_t arrSize)
{
	//Convert the world ray into the local coordinate system of the scene
	Ray sceneRay = originalRay.convertRayToLocalSpace(sceneInfo->translationVector, sceneInfo->scale);

	//determine the region coordinates of the ray
	uint32_t regionX = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getX() / BLOCK_SIZE));
	uint32_t regionY = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getY() / BLOCK_SIZE));
	uint32_t regionZ = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getZ() / BLOCK_SIZE));

	//Transform the ray into the local coordinates of the current region it is located in
	Ray localRay = sceneRay.convertRayToLocalSpace(Vector3f(regionX * BLOCK_SIZE, regionY * BLOCK_SIZE, regionZ * BLOCK_SIZE), 1.0f);

	//while the region coordinates are within the bounds of the scene, keep doing calculation
	while (isRayInScene(regionX, regionY, regionZ))
	{
		StorageStructure* regionStorageStructure = voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize];
		while (regionStorageStructure == nullptr)
		{
			float nextX = localRay.getDirection().getX() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
			float nextY = localRay.getDirection().getY() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
			float nextZ = localRay.getDirection().getZ() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;

			float tX = (nextX - localRay.getOrigin().getX()) / localRay.getDirection().getX();
			float tY = (nextY - localRay.getOrigin().getY()) / localRay.getDirection().getY();
			float tZ = (nextZ - localRay.getOrigin().getZ()) / localRay.getDirection().getZ();
			//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
			float tMin = min(tX, min(tY, tZ));

			localRay = Ray(localRay.getOrigin() + tMin * localRay.getDirection(), localRay.getDirection());
			//Find the difference of the next region coordinates based on the processed ray from the previous ray marching algorithm
			int32_t regionXDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getX() / BLOCK_SIZE));
			int32_t regionYDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getY() / BLOCK_SIZE));
			int32_t regionZDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getZ() / BLOCK_SIZE));
			//Update the region coordiantes to indicate the current region
			regionX += regionXDiff;
			regionY += regionYDiff;
			regionZ += regionZDiff;
			//Update the localRay to be in the space of the new region
			localRay = localRay.convertRayToLocalSpace(Vector3f(regionXDiff * BLOCK_SIZE, regionYDiff * BLOCK_SIZE, regionZDiff * BLOCK_SIZE), 1.0f);
			if (isRayInScene(regionX, regionY, regionZ))
				regionStorageStructure = voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize];
			else return 0;
		}

		//Vector is passed along for lighting calculations
		Vector3f regionWorldPosition = sceneInfo->translationVector + Vector3f(regionX * BLOCK_SIZE, regionY * BLOCK_SIZE, regionZ * BLOCK_SIZE);
		//call the raymarching function for that voxel structure
		uint32_t voxelColor = rayMarchVoxelGrid(localRay, regionWorldPosition, voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize]);
		if (voxelColor != EMPTY_VAL)
		{
			return voxelColor;
		}
		//Find the difference of the next region coordinates based on the processed ray from the previous ray marching algorithm
		int32_t regionXDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getX() / BLOCK_SIZE));
		int32_t regionYDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getY() / BLOCK_SIZE));
		int32_t regionZDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getZ() / BLOCK_SIZE));
		//Update the region coordiantes to indicate the current region
		regionX += regionXDiff;
		regionY += regionYDiff;
		regionZ += regionZDiff;
		//Update the localRay to be in the space of the new region
		localRay = localRay.convertRayToLocalSpace(Vector3f(regionXDiff * BLOCK_SIZE, regionYDiff * BLOCK_SIZE, regionZDiff * BLOCK_SIZE), 1.0f);
	}

	//Return the background color
	return 0;
}

__device__ Ray2D generateLongestDirectionRay2D(Ray ray, uint32_t& longestAxis, uint32_t& middleAxis, uint32_t& shorestAxis)
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
			middleAxis = 1;
			shorestAxis = 2;
		}
		else
		{
			middleAxis = 2;
			shorestAxis = 1;
		}
		constant = 1.0f / xDirAbs;
	}
	else if (yDirAbs > zDirAbs)
	{
		longestAxis = 1;
		if (xDirAbs > zDirAbs)
		{
			middleAxis = 0;
			shorestAxis = 2;
		}
		else
		{
			middleAxis = 2;
			shorestAxis = 0;
		}
		constant = 1.0f / yDirAbs;
	}
	else
	{
		longestAxis = 2;
		if (xDirAbs > yDirAbs)
		{
			middleAxis = 0;
			shorestAxis = 1;
		}
		else
		{
			middleAxis = 1;
			shorestAxis = 0;
		}
		constant = 1.0f / zDirAbs;
	}
	return Ray2D(
		Vector2f(ray.getOrigin()[middleAxis], ray.getOrigin()[shorestAxis]), 
		Vector2f(ray.getDirection()[middleAxis] * constant, ray.getDirection()[shorestAxis] * constant)
	);
}

__device__ __forceinline__ bool areGridValuesInRegion(uint32_t x, uint32_t y, uint32_t z)
{
	return x < BLOCK_SIZE && y < BLOCK_SIZE && z < BLOCK_SIZE;
}

__device__ uint32_t performVoxelSpaceJump(StorageStructure* storageStructure, Ray& oldRay, Ray& ray, int32_t* gridValues, int32_t* axisDiff, 
	uint32_t longestAxis, uint32_t middleAxis, uint32_t shortestAxis)
{
	float tMin = 0.0f;
	while (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
	{
		//Calculate the next block locations for each axis
		int32_t nextX = oldRay.getDirection().getX() > 0.0f ? ((gridValues[0] / 8) + 1) * 8 : (gridValues[0] / 8) * 8 - 1;
		int32_t nextY = oldRay.getDirection().getY() > 0.0f ? ((gridValues[1] / 8) + 1) * 8 : (gridValues[1] / 8) * 8 - 1;
		int32_t nextZ = oldRay.getDirection().getZ() > 0.0f ? ((gridValues[2] / 8) + 1) * 8 : (gridValues[2] / 8) * 8 - 1;

		//Calculate the t-values for the shortest and middle axes
		float tX = (nextX - oldRay.getOrigin().getX()) / oldRay.getDirection().getX();
		float tY = (nextY - oldRay.getOrigin().getY()) / oldRay.getDirection().getY();
		float tZ = (nextZ - oldRay.getOrigin().getZ()) / oldRay.getDirection().getZ();
		tMin = min(tX, min(tY, tZ));

		oldRay = Ray(oldRay.getOrigin() + tMin * oldRay.getDirection(), oldRay.getDirection());
		gridValues[0] = static_cast<int32_t>(oldRay.getOrigin().getX());
		gridValues[1] = static_cast<int32_t>(oldRay.getOrigin().getY());
		gridValues[2] = static_cast<int32_t>(oldRay.getOrigin().getZ());

		if (!areGridValuesInRegion(gridValues[0], gridValues[1], gridValues[2]))
		{
			return 0;//EMPTY_VAL;
		}
	}

	//Check the voxel location
	uint32_t voxelColor = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
	if (voxelColor != EMPTY_VAL)
	{
		//TODO add lighting calculation stuff
		return voxelColor;
	}

	//Calculate the t-value for the next location of the ray which is snapped to the longest axis
	float tNext = std::ceilf(tMin) - tMin;
	ray = Ray(oldRay.getOrigin() + tNext * oldRay.getDirection(), oldRay.getDirection());
	//Caclulate middle and shortest axes voxel coordinate differences
	axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
	axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];

	//Continue to the next iteration to avoid old axis diffs being applied
	return CONTINUE_VAL;
}

__device__ uint32_t rayMarchVoxelGridLongestAxis(Ray& originalRay, const Vector3f regionWorldPosition, StorageStructure* storageStructure)
{
	//Both -1.0f and 1.0f can be represented correctly so when orginally snapping to the grid an epsilon needs to be employed and will keep things the correct way
	uint32_t longestAxis;
	uint32_t middleAxis;
	uint32_t shortestAxis;

	//Create the longest axis ray and populate the longestAxis, middleAxis, shortestAxis indices
	Ray oldRay = originalRay.convertRayToLongestAxisDirection(originalRay, longestAxis, middleAxis, shortestAxis);
	int32_t gridValues[3] = { static_cast<int32_t>(originalRay.getOrigin().getX()),
							 static_cast<int32_t>(originalRay.getOrigin().getY()),
							 static_cast<int32_t>(originalRay.getOrigin().getZ()) };
	int32_t axisDiff[3] = { 0, 0, 0 };
	axisDiff[longestAxis] = originalRay.getDirection()[longestAxis] < 0.0f ? -1 : 1;

	//Snap the longest direction vector axis to the grid
	float t = axisDiff[longestAxis] > 0 ?
		(gridValues[longestAxis] + EPSILON + 1 - originalRay.getOrigin()[longestAxis]) / axisDiff[longestAxis] :
		(gridValues[longestAxis] - EPSILON - originalRay.getOrigin()[longestAxis]) / axisDiff[longestAxis];
	Ray ray = Ray(oldRay.getOrigin() + oldRay.getDirection() * t, oldRay.getDirection());
	//Caclulate middle and shortest axes voxel coordinate differences
	axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
	axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
	//Check if the ray's middle axis is moving in the positive or negative direction to assign the correct conversion function
	float (*decimalToIntFunc)(float) = ray.getDirection()[middleAxis] < 0.0f ? &std::floorf : &std::ceilf;

	//Check to make sure the next grid values are within the region
	while (areGridValuesInRegion(gridValues[longestAxis] + axisDiff[longestAxis], 
		gridValues[middleAxis] + axisDiff[middleAxis], 
		gridValues[shortestAxis] + axisDiff[shortestAxis]))
	{
		//Both the shortest and middle axis have axis differences
		if (axisDiff[shortestAxis] != 0 && axisDiff[middleAxis] != 0)
		{
			//Find the t-value where the middle axis ray would intersect that axis
			float t1 = ((*decimalToIntFunc)((oldRay.getOrigin()[middleAxis])) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis];
			//Cacluate the position of the shortest ray at that t-value location for the middle axis
			float shortestPosition = oldRay.getOrigin()[shortestAxis] + oldRay.getDirection()[shortestAxis] * t1;
			//Check to see if there is an axis diff in the shorter axis - if there is apply the shorter axis first
			int32_t shorterDiff = static_cast<int32_t>(floorf(shortestPosition)) - gridValues[shortestAxis];
			uint32_t applyOrder[2] = { middleAxis, shortestAxis };
			if (shorterDiff != 0)
			{
				applyOrder[0] = shortestAxis;
				applyOrder[1] = middleAxis;
			}

			gridValues[applyOrder[0]] += axisDiff[applyOrder[0]];
			if (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
			{
				uint32_t jumpResult = performVoxelSpaceJump(storageStructure, oldRay, ray, gridValues, axisDiff, longestAxis, middleAxis, shortestAxis);
				if (jumpResult != CONTINUE_VAL)
					return jumpResult;
				continue;
			}
			uint32_t colorValue = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
			if (colorValue != EMPTY_VAL)
			{
				Vector3f normal = Vector3f(0.0f, 0.0f, 0.0f);
				normal[applyOrder[0]] = std::copysignf(1.0f, -ray.getDirection()[applyOrder[0]]);
				if (USE_POINT_LIGHT)
				{
					Vector3f hitPoint = getHitLocation(regionWorldPosition, Vector3f(0.0f, 0.0f, 0.0f));
					return applyPointLightingToColor(colorValue, hitPoint, normal);
				}
				return applyDirectionalLightingToColor(colorValue, normal);
			}

			gridValues[applyOrder[1]] += axisDiff[applyOrder[1]];
			if (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
			{
				uint32_t jumpResult = performVoxelSpaceJump(storageStructure, oldRay, ray, gridValues, axisDiff, longestAxis, middleAxis, shortestAxis);
				if (jumpResult != CONTINUE_VAL)
					return jumpResult;
				continue;
			}
			colorValue = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
			if (colorValue != EMPTY_VAL)
			{
				Vector3f normal = Vector3f(0.0f, 0.0f, 0.0f);
				normal[applyOrder[1]] = std::copysignf(1.0f, -ray.getDirection()[applyOrder[1]]);
				if (USE_POINT_LIGHT)
				{
					Vector3f hitPoint = getHitLocation(regionWorldPosition, Vector3f(0.0f, 0.0f, 0.0f));
					return applyPointLightingToColor(colorValue, hitPoint, normal);
				}
				return applyDirectionalLightingToColor(colorValue, normal);
			}
		}
		//Only middle axis has an axis difference
		else if (axisDiff[middleAxis] != 0)
		{
			gridValues[middleAxis] += axisDiff[middleAxis];
			if (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
			{
				uint32_t jumpResult = performVoxelSpaceJump(storageStructure, oldRay, ray, gridValues, axisDiff, longestAxis, middleAxis, shortestAxis);
				if (jumpResult != CONTINUE_VAL)
					return jumpResult;
				continue;
			}
			uint32_t colorValue = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
			if (colorValue != EMPTY_VAL)
			{
				Vector3f normal = Vector3f(0.0f, 0.0f, 0.0f);
				normal[middleAxis] = std::copysignf(1.0f, -ray.getDirection()[middleAxis]);
				if (USE_POINT_LIGHT)
				{
					Vector3f hitPoint = getHitLocation(regionWorldPosition, Vector3f(0.0f, 0.0f, 0.0f));
					return applyPointLightingToColor(colorValue, hitPoint, normal);
				}
				return applyDirectionalLightingToColor(colorValue, normal);
			}
		}
		//Only shortest axis has an axis difference
		else if (axisDiff[shortestAxis] != 0)
		{
			gridValues[shortestAxis] += axisDiff[shortestAxis];
			if (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
			{
				uint32_t jumpResult = performVoxelSpaceJump(storageStructure, oldRay, ray, gridValues, axisDiff, longestAxis, middleAxis, shortestAxis);
				if (jumpResult != CONTINUE_VAL)
					return jumpResult;
				continue;
			}
			uint32_t colorValue = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
			if (colorValue != EMPTY_VAL)
			{
				Vector3f normal = Vector3f(0.0f, 0.0f, 0.0f);
				normal[shortestAxis] = std::copysignf(1.0f, -ray.getDirection()[shortestAxis]);
				if (USE_POINT_LIGHT)
				{
					Vector3f hitPoint = getHitLocation(regionWorldPosition, Vector3f(0.0f, 0.0f, 0.0f));
					return applyPointLightingToColor(colorValue, hitPoint, normal);
				}
				return applyDirectionalLightingToColor(colorValue, normal);
			}
		}
		//Always perform longest axis check
		gridValues[longestAxis] += axisDiff[longestAxis];
		if (!storageStructure->doesVoxelSpaceExist(gridValues[0], gridValues[1], gridValues[2]))
		{
			uint32_t jumpResult = performVoxelSpaceJump(storageStructure, oldRay, ray, gridValues, axisDiff, longestAxis, middleAxis, shortestAxis);
			if (jumpResult != CONTINUE_VAL)
				return jumpResult;
			continue;
		}
		uint32_t colorValue = storageStructure->lookupVoxel(gridValues[0], gridValues[1], gridValues[2]);
		if (colorValue != EMPTY_VAL)
		{
			Vector3f normal = Vector3f(0.0f, 0.0f, 0.0f);
			normal[longestAxis] = std::copysignf(1.0f, -ray.getDirection()[longestAxis]);
			if (USE_POINT_LIGHT)
			{
				Vector3f hitPoint = getHitLocation(regionWorldPosition, Vector3f(0.0f, 0.0f, 0.0f));
				return applyPointLightingToColor(colorValue, hitPoint, normal);
			}
			return applyDirectionalLightingToColor(colorValue, normal);
		}
		
		oldRay = ray;
		//Set the ray to the next location which is just moving the ray by 1 unit of its direction vector (moving 1 voxel unit along longest axis)
		ray = Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection());
		//Caclulate middle and shortest axes voxel coordinate differences
		axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
		axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
	}

	//Ray is now outside of the bounds of region after performing its last jump so we need to use the origin from the old ray
	originalRay = Ray(oldRay.getOrigin(), originalRay.getDirection());
	//Peforming the original ray marching algorithm will update the ray position correctly
	return rayMarchVoxelGrid(originalRay, regionWorldPosition, storageStructure);
}

__device__ uint32_t rayMarchVoxelSceneLongestAxis(const Ray& originalRay, const VoxelSceneInfo* sceneInfo, StorageStructure** voxelScene, uint32_t arrSize)
{
	//Convert the world ray into the local coordinate system of the scene
	Ray sceneRay = originalRay.convertRayToLocalSpace(sceneInfo->translationVector, sceneInfo->scale);

	//determine the region coordinates of the ray
	uint32_t regionX = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getX() / BLOCK_SIZE));
	uint32_t regionY = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getY() / BLOCK_SIZE));
	uint32_t regionZ = static_cast<uint32_t>(floorf(sceneRay.getOrigin().getZ() / BLOCK_SIZE));

	//Transform the ray into the local coordinates of the current region it is located in
	Ray localRay = sceneRay.convertRayToLocalSpace(Vector3f(regionX * BLOCK_SIZE, regionY * BLOCK_SIZE, regionZ * BLOCK_SIZE), 1.0f);

	//while the region coordinates are within the bounds of the scene, keep doing calculation
	while (isRayInScene(regionX, regionY, regionZ))
	{
		StorageStructure* regionStorageStructure = voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize];
		while (regionStorageStructure == nullptr)
		{
			float nextX = localRay.getDirection().getX() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
			float nextY = localRay.getDirection().getY() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;
			float nextZ = localRay.getDirection().getZ() > 0.0f ? BLOCK_SIZE + EPSILON : 0.0f - EPSILON;

			float tX = (nextX - localRay.getOrigin().getX()) / localRay.getDirection().getX();
			float tY = (nextY - localRay.getOrigin().getY()) / localRay.getDirection().getY();
			float tZ = (nextZ - localRay.getOrigin().getZ()) / localRay.getDirection().getZ();
			//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
			float tMin = min(tX, min(tY, tZ));

			localRay = Ray(localRay.getOrigin() + tMin * localRay.getDirection(), localRay.getDirection());
			//Find the difference of the next region coordinates based on the processed ray from the previous ray marching algorithm
			int32_t regionXDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getX() / BLOCK_SIZE));
			int32_t regionYDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getY() / BLOCK_SIZE));
			int32_t regionZDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getZ() / BLOCK_SIZE));
			//Update the region coordiantes to indicate the current region
			regionX += regionXDiff;
			regionY += regionYDiff;
			regionZ += regionZDiff;
			//Update the localRay to be in the space of the new region
			localRay = localRay.convertRayToLocalSpace(Vector3f(regionXDiff * BLOCK_SIZE, regionYDiff * BLOCK_SIZE, regionZDiff * BLOCK_SIZE), 1.0f);
			if (isRayInScene(regionX, regionY, regionZ))
				regionStorageStructure = voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize];
			else return 0;
		}

		//Vector is passed along for lighting calculations
		Vector3f regionWorldPosition = sceneInfo->translationVector + Vector3f(regionX * BLOCK_SIZE, regionY * BLOCK_SIZE, regionZ * BLOCK_SIZE);
		//call the raymarching function for that voxel structure
		uint32_t voxelColor = rayMarchVoxelGridLongestAxis(localRay, regionWorldPosition, voxelScene[regionX + regionY * arrSize + regionZ * arrSize * arrSize]);
		if (voxelColor != EMPTY_VAL)
		{
			return voxelColor;
		}
		//Find the difference of the next region coordinates based on the processed ray from the previous ray marching algorithm
		int32_t regionXDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getX() / BLOCK_SIZE));
		int32_t regionYDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getY() / BLOCK_SIZE));
		int32_t regionZDiff = static_cast<int32_t>(floorf(localRay.getOrigin().getZ() / BLOCK_SIZE));
		//Update the region coordiantes to indicate the current region
		regionX += regionXDiff;
		regionY += regionYDiff;
		regionZ += regionZDiff;
		//Update the localRay to be in the space of the new region
		localRay = localRay.convertRayToLocalSpace(Vector3f(regionXDiff * BLOCK_SIZE, regionYDiff * BLOCK_SIZE, regionZDiff * BLOCK_SIZE), 1.0f);
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

__forceinline__ __device__ void populateStorageStructures(void** storageStructureArr, uint32_t arrSize, StorageStructure** storageArr, StorageType storageType)
{
	switch (storageType)
	{
	case StorageType::HASH_TABLE:
		for (uint32_t i = 0; i < arrSize; i++)
		{
			if(storageStructureArr[i])
				storageArr[i] = new HashTableStorageStructure(static_cast<CuckooHashTable*>(storageStructureArr[i]));
		}
		break;
	case StorageType::VOXEL_CLUSTER_STORE:
		for (uint32_t i = 0; i < arrSize; i++)
		{
			if(storageStructureArr[i])
				storageArr[i] = new VCSStorageStructure(static_cast<VoxelClusterStore*>(storageStructureArr[i]));
		}
		break;
	}
}

__global__ void rayMarchSceneOriginal(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelSceneInfo* voxelSceneInfo, uint8_t* framebuffer, 
	StorageStructure** voxelScene, uint32_t arrSize, StorageType storageType)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray worldRay = calculateWorldRay(xPixel, yPixel, imgWidth, imgHeight, camera);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelScene(worldRay, voxelSceneInfo, voxelScene, arrSize);

	writeColorToFramebuffer(xPixel, yPixel, imgWidth, color, framebuffer);
}

__global__ void rayMarchSceneJumpAxis(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelSceneInfo* voxelSceneInfo, uint8_t* framebuffer,
	StorageStructure** voxelScene, uint32_t arrSize, StorageType storageType)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray worldRay = calculateWorldRay(xPixel, yPixel, imgWidth, imgHeight, camera);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelSceneLongestAxis(worldRay, voxelSceneInfo, voxelScene, arrSize);

	writeColorToFramebuffer(xPixel, yPixel, imgWidth, color, framebuffer);
}

__global__ void generateVoxelScene(StorageStructure** devicePtr, void** storageStructureArr, uint32_t arrSize, StorageType storageType)
{
	populateStorageStructures(storageStructureArr, arrSize, devicePtr, storageType);
}