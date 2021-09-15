#pragma once

#include <cuda_runtime.h>

#include "VoxelStructure.cuh"
#include "camera/Camera.cuh"
#include "rays/Ray.cuh"


#include "../geometry/VoxelFunctions.cuh"

#include "../storage/VoxelClusterStore.cuh"
#include "../storage/CuckooHashTable.cuh"

__device__ float applyCeilAndPosEpsilon(float input)
{
	return ceilf(input) + EPSILON;
}

__device__ float applyFloorAndNegEpsilon(float input)
{
	return floorf(input) - EPSILON;
}

//Raymarch Original w/ VCS
__device__ uint32_t rayMarchVoxelGridVCS(const Ray& originalRay, const VoxelStructure* voxelStructure, const VoxelClusterStore* voxelClusterStore)
{
	Ray ray = originalRay;
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;

	while (voxelStructure->isRayInStructure(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t gridValues[] =
		{
			static_cast<int32_t>(ray.getOrigin().getX()),
			static_cast<int32_t>(ray.getOrigin().getY()),
			static_cast<int32_t>(ray.getOrigin().getZ())
		};

		//Check if the voxel is in the map
		uint32_t voxelColor = voxelClusterStore->lookupVoxelNoFinishVal(gridValues, ray);
		if (voxelColor != EMPTY_KEY)
		{
			return voxelColor;
		}

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
	}

	//Return the background color of black
	return 0;
}

//Raymarch Original w/ VoxelHashTable
__device__ uint32_t rayMarchVoxelGridHashTable(const Ray& originalRay, const VoxelStructure* voxelStructure, const CuckooHashTable* hashTable)
{
	Ray ray = originalRay;
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon : applyFloorAndNegEpsilon;

	while (voxelStructure->isRayInStructure(ray))
	{
		//Perform the lookup first so that the next ray location can be checked before lookup to avoid accessing memory that should not be in VCS
		int32_t gridValues[] =
		{
			static_cast<int32_t>(ray.getOrigin().getX()),
			static_cast<int32_t>(ray.getOrigin().getY()),
			static_cast<int32_t>(ray.getOrigin().getZ())
		};

		//Check if the voxel is in the map
		uint32_t voxelColor = hashTable->lookupVoxel(gridValues, ray);
		if (voxelColor != EMPTY_KEY)
		{
			return voxelColor;
		}

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
	}

	//Return the background color of black
	return 0;
}

//Raymarch Jump Axis w/ VoxelHashTable
__device__ uint32_t checkRayJumpForVoxelsHashTable(Ray& oldRay, Ray& ray, float (*decimalToIntFunc)(float), const CuckooHashTable* hashTable, int32_t* axisDiff, int32_t* gridValues,
	uint32_t shortestAxis, uint32_t middleAxis, uint32_t longestAxis, bool shortestCheck, bool middleCheck, bool longestCheck)
{
	if (shortestCheck && middleCheck && axisDiff[middleAxis] != 0 && axisDiff[shortestAxis] != 0)
	{
		float t1 = (*decimalToIntFunc)(((oldRay.getOrigin()[middleAxis]) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis]);
		float shortestPosition = oldRay.getOrigin()[shortestAxis] + oldRay.getDirection()[shortestAxis] * t1;
		int32_t shorterDiff = static_cast<int32_t>(floorf(shortestPosition)) - gridValues[shortestAxis];
		uint32_t applyOrder[2] = { middleAxis, shortestAxis };
		if (shorterDiff != 0)
		{
			applyOrder[0] = shortestAxis;
			applyOrder[1] = middleAxis;
		}
		//apply shorter first
		gridValues[applyOrder[0]] += axisDiff[applyOrder[0]];
		uint32_t colorValue = hashTable->lookupVoxel(gridValues, ray);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
		//apply longer second
		gridValues[applyOrder[1]] += axisDiff[applyOrder[1]];
		colorValue = hashTable->lookupVoxel(gridValues, ray);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}
	else if (middleCheck && axisDiff[middleAxis] != 0)
	{
		gridValues[middleAxis] += axisDiff[middleAxis];
		uint32_t colorValue = hashTable->lookupVoxel(gridValues, ray);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}
	else if (shortestCheck && axisDiff[shortestAxis] != 0)
	{
		gridValues[shortestAxis] += axisDiff[shortestAxis];
		uint32_t colorValue = hashTable->lookupVoxel(gridValues, ray);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	if (longestCheck)
	{
		gridValues[longestAxis] += axisDiff[longestAxis];
		uint32_t colorValue = hashTable->lookupVoxel(gridValues, ray);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	return EMPTY_VAL;
}

__device__ uint32_t rayMarchVoxelGridLongestAxisHashTable(const Ray& originalRay, const VoxelStructure* voxelStructure, const CuckooHashTable* hashTable)
{
	uint32_t longestAxis;
	uint32_t middleAxis;
	uint32_t shortestAxis;
	Ray oldRay = originalRay.convertRayToLongestAxisDirection(originalRay, longestAxis, middleAxis, shortestAxis);

	int32_t gridValues[3] = { static_cast<int32_t>(oldRay.getOrigin().getX()), static_cast<int32_t>(oldRay.getOrigin().getY()), static_cast<int32_t>(oldRay.getOrigin().getZ()) };
	int32_t axisDiff[3] = { 0, 0, 0 };
	axisDiff[longestAxis] = oldRay.getDirection()[longestAxis] < 0.0f ? -1 : 1;

	//Snap the longest direction vector axis to the grid first
	float t = oldRay.getDirection()[longestAxis] > 0.0f ?
		(gridValues[longestAxis] + EPSILON + 1 - oldRay.getOrigin()[longestAxis]) / oldRay.getDirection()[longestAxis] :
		(gridValues[longestAxis] - EPSILON - oldRay.getOrigin()[longestAxis]) / oldRay.getDirection()[longestAxis];
	Ray ray = Ray(oldRay.getOrigin() + oldRay.getDirection() * t, oldRay.getDirection());
	//Calculate the other axis (besides the longest axis) voxel coordinate differences
	axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
	axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
	//Check if the ray's middle axis is moving in the positive or negative direction to assign the correct conversion function
	float (*decimalToIntFunc)(float) = ray.getDirection()[middleAxis] < 0.0f ? &std::floorf : &std::ceilf;

	//Perform a check on all voxels in the axis jump to check if any voxels were intersected
	uint32_t colorValue = checkRayJumpForVoxelsHashTable(oldRay, ray, decimalToIntFunc, hashTable, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis, true, true, true);
	if (colorValue != EMPTY_VAL)
	{
		return colorValue;
	}

	//Loop until the next ray location is outside of the voxel grid
	while (voxelStructure->isRayInStructure(Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection())))
	{
		Ray oldRay = ray;
		ray = Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection());
		axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
		axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
		colorValue = checkRayJumpForVoxelsHashTable(oldRay, ray, decimalToIntFunc, hashTable, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis, true, true, true);
		if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	//Check if anymore voxels need to be checked while applying a restriction that the voxel must be inside the grid boundaries
	oldRay = ray;
	ray = Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection());
	axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
	axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
	uint32_t nextMidAxis = gridValues[middleAxis] + axisDiff[middleAxis];
	uint32_t nextShortAxis = gridValues[shortestAxis] + axisDiff[shortestAxis];
	uint32_t nextLongAxis = gridValues[longestAxis] + axisDiff[longestAxis];
	colorValue = checkRayJumpForVoxelsHashTable(oldRay, ray, decimalToIntFunc, hashTable, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis,
		nextShortAxis < BLOCK_SIZE, nextMidAxis < BLOCK_SIZE, nextLongAxis < BLOCK_SIZE);
	if (colorValue != EMPTY_VAL)
	{
		return colorValue;
	}

	//Return a background color of black if no voxel is hit by the ray
	return 0;
}

__forceinline__ __device__ Ray calculateLocalRay1(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t imgHeight, const Camera* camera, const VoxelStructure* voxelStructure)
{
	//Calculate normalized camera space coordinates (between 0 and 1) of the center of the pixel
	float uCamera = (xPixel + 0.5f) / static_cast<float>(imgWidth);
	//Since framebuffer coordinates have posative y going down and 3D position has a positive y going up. The imgHeight needs to be subracted to make up for this difference
	float vCamera = (imgHeight - yPixel + 0.5f) / static_cast<float>(imgHeight);

	//Calculate the ray cast by the camera with the camera space coordiantes for the pixel
	Ray ray = camera->generateRay(uCamera, vCamera);

	//Create the local ray which is in the coordinate domain of the voxel structure it will be traversing
	return ray.convertRayToLocalSpace(voxelStructure->translationVector, voxelStructure->scale);
}

__forceinline__ __device__ void writeColorToFramebuffer1(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t color, uint8_t* framebuffer)
{
	//Set the framebuffer location for that pixel to the returned color
	uint32_t pixelIndex = yPixel * imgWidth * 3 + xPixel * 3;
	framebuffer[pixelIndex] = voxelfunc::getRedComponent(color);
	framebuffer[pixelIndex + 1] = voxelfunc::getGreenComponent(color);
	framebuffer[pixelIndex + 2] = voxelfunc::getBlueComponent(color);
}

__global__ void rayMarchSceneOriginalVCS(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer,
	VoxelClusterStore* voxelClusterStorePtr)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray localRay = calculateLocalRay1(xPixel, yPixel, imgWidth, imgHeight, camera, voxelStructure);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGridVCS(localRay, voxelStructure, voxelClusterStorePtr);

	writeColorToFramebuffer1(xPixel, yPixel, imgWidth, color, framebuffer);
}

__global__ void rayMarchSceneOriginalHashTable(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer,
	CuckooHashTable* hashTablePtr)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray localRay = calculateLocalRay1(xPixel, yPixel, imgWidth, imgHeight, camera, voxelStructure);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGridHashTable(localRay, voxelStructure, hashTablePtr);

	writeColorToFramebuffer1(xPixel, yPixel, imgWidth, color, framebuffer);
}

__global__ void rayMarchSceneJumpAxisHashTable(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer,
	CuckooHashTable* hashTablePtr)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	Ray localRay = calculateLocalRay1(xPixel, yPixel, imgWidth, imgHeight, camera, voxelStructure);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGridLongestAxisHashTable(localRay, voxelStructure, hashTablePtr);

	writeColorToFramebuffer1(xPixel, yPixel, imgWidth, color, framebuffer);
}