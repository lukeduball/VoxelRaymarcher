#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <math.h>

#include "VoxelStructure.cuh"
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

__device__ uint32_t rayMarchVoxelGrid(const Ray& originalRay, const VoxelStructure* voxelStructure, const StorageStructure* storageStructure)
{
	Ray ray = originalRay;
	//Calculate once outside of the loop to increase performance
	float (*nextXFunc)(float) = ray.getDirection().getX() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextYFunc)(float) = ray.getDirection().getY() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;
	float (*nextZFunc)(float) = ray.getDirection().getZ() > 0.0f ? applyCeilAndPosEpsilon1 : applyFloorAndNegEpsilon1;

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
		uint32_t voxelColor = storageStructure->lookupVoxel(gridValues, ray);
		if (voxelColor != EMPTY_KEY && voxelColor != FINISH_VAL)
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

__device__ uint32_t checkRayJumpForVoxels(Ray& oldRay, Ray& ray, float (*decimalToIntFunc)(float),
	const StorageStructure* storageStructure, int32_t* axisDiff, int32_t* gridValues, 
	uint32_t shortestAxis, uint32_t middleAxis, uint32_t longestAxis, bool shortCheck, bool middleCheck, bool longestCheck)
{
	if (shortCheck && middleCheck && axisDiff[middleAxis] != 0 && axisDiff[shortestAxis] != 0)
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
		uint32_t colorValue = storageStructure->lookupVoxel(gridValues, ray);
		if (colorValue == FINISH_VAL)
		{
			return FINISH_VAL;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
		//apply longer second
		gridValues[applyOrder[1]] += axisDiff[applyOrder[1]];
		colorValue = storageStructure->lookupVoxel(gridValues, ray);
		if (colorValue == FINISH_VAL)
		{
			return FINISH_VAL;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}
	else if (middleCheck && axisDiff[middleAxis] != 0)
	{
		gridValues[middleAxis] += axisDiff[middleAxis];
		uint32_t colorValue = storageStructure->lookupVoxel(gridValues, ray);
		if (colorValue == FINISH_VAL)
		{
			return FINISH_VAL;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}
	else if (shortCheck && axisDiff[shortestAxis] != 0)
	{
		gridValues[shortestAxis] += axisDiff[shortestAxis];
		uint32_t colorValue = storageStructure->lookupVoxel(gridValues, ray);
		if (colorValue == FINISH_VAL)
		{
			return FINISH_VAL;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	if (longestCheck)
	{
		gridValues[longestAxis] += axisDiff[longestAxis];
		uint32_t colorValue = storageStructure->lookupVoxel(gridValues, ray);
		if (colorValue == FINISH_VAL)
		{
			return FINISH_VAL;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	return EMPTY_VAL;
}

__device__ uint32_t rayMarchVoxelGridAxisJump(const Ray& originalRay, const VoxelStructure* voxelStructure, const StorageStructure* storageStructure)
{
	//Both -1.0f and 1.0f can be represented correctly so when orginally snapping to the grid an epsilon needs to be employed and will keep things the correct way
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
	uint32_t colorValue = checkRayJumpForVoxels(oldRay, ray, decimalToIntFunc, storageStructure, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis, true, true, true);
	if (colorValue == FINISH_VAL)
	{
		return 0;
	}
	else if (colorValue != EMPTY_VAL)
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
		colorValue = checkRayJumpForVoxels(oldRay, ray, decimalToIntFunc, storageStructure, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis, true, true, true);
		if (colorValue == FINISH_VAL)
		{
			return 0;
		}
		else if (colorValue != EMPTY_VAL)
		{
			return colorValue;
		}
	}

	//TODO -- Needs a functional test to be checked on axis boundaries in the future
	//Check if anymore voxels need to be checked while applying a restiction that the voxel must be inside the grid boundaries
	oldRay = ray;
	ray = Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection());
	axisDiff[middleAxis] = static_cast<int32_t>(ray.getOrigin()[middleAxis]) - gridValues[middleAxis];
	axisDiff[shortestAxis] = static_cast<int32_t>(ray.getOrigin()[shortestAxis]) - gridValues[shortestAxis];
	uint32_t nextMidAxis = gridValues[middleAxis] + axisDiff[middleAxis];
	uint32_t nextShortAxis = gridValues[shortestAxis] + axisDiff[shortestAxis];
	uint32_t nextLongAxis = gridValues[longestAxis] + axisDiff[longestAxis];
	colorValue = checkRayJumpForVoxels(oldRay, ray, decimalToIntFunc, storageStructure, axisDiff, gridValues, shortestAxis, middleAxis, longestAxis,
		nextShortAxis < BLOCK_SIZE, nextMidAxis < BLOCK_SIZE, nextLongAxis < BLOCK_SIZE);
	if (colorValue == FINISH_VAL)
	{
		return 0;
	}
	else if (colorValue != EMPTY_VAL)
	{
		return colorValue;
	}
	
	//Return a background color of black if no voxel is hit by the ray
	return 0;
}

__forceinline__ __device__ Ray calculateLocalRay(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t imgHeight, const Camera* camera, const VoxelStructure* voxelStructure)
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

__forceinline__ __device__ void writeColorToFramebuffer(uint32_t xPixel, uint32_t yPixel, uint32_t imgWidth, uint32_t color, uint8_t* framebuffer)
{
	//Set the framebuffer location for that pixel to the returned color
	uint32_t pixelIndex = yPixel * imgWidth * 3 + xPixel * 3;
	framebuffer[pixelIndex] = color >> 16;
	framebuffer[pixelIndex + 1] = (color >> 8) & 0xFF;
	framebuffer[pixelIndex + 2] = color & 0xFF;
}

__forceinline__ __device__ StorageStructure* getStorageStructure(VoxelClusterStore* voxelClusterStorePtr, CuckooHashTable* cuckooHashTablePtr)
{
	if (voxelClusterStorePtr != nullptr)
	{
		return new VCSStorageStructure(voxelClusterStorePtr);
	}
	return new HashTableStorageStructure(cuckooHashTablePtr);
}

__global__ void rayMarchSceneOriginal(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer, 
	VoxelClusterStore* voxelClusterStorePtr, CuckooHashTable* cuckooHashTablePtr)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	//Setup the correct storage structure
	StorageStructure* storageStructure = getStorageStructure(voxelClusterStorePtr, cuckooHashTablePtr);

	Ray localRay = calculateLocalRay(xPixel, yPixel, imgWidth, imgHeight, camera, voxelStructure);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGrid(localRay, voxelStructure, storageStructure);

	writeColorToFramebuffer(xPixel, yPixel, imgWidth, color, framebuffer);

	delete storageStructure;
}

__global__ void rayMarchSceneJumpAxis(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer,
	VoxelClusterStore* voxelClusterStorePtr, CuckooHashTable* cuckooHashTablePtr)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	//Setup the correct storage structure
	StorageStructure* storageStructure = getStorageStructure(voxelClusterStorePtr, cuckooHashTablePtr);

	Ray localRay = calculateLocalRay(xPixel, yPixel, imgWidth, imgHeight, camera, voxelStructure);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGridAxisJump(localRay, voxelStructure, storageStructure);

	writeColorToFramebuffer(xPixel, yPixel, imgWidth, color, framebuffer);

	delete storageStructure;
}