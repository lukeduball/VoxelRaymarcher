#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <math.h>

#include "VoxelStructure.cuh"
#include "camera/Camera.cuh"
#include "rays/Ray.cuh"

//Start with 1920x1080 HD image
//Split up the image into 30x30 sections - the GCD is 120 (only 1024 threads allowed per block)

__device__ int generate3DInteger(int x, int y, int z)
{
	return (x << 20) | (y << 10) | z;
}

constexpr float EPSILON = 0.0001f;

__device__ uint32_t rayMarchVoxelGrid(const Ray& originalRay, const VoxelStructure* voxelStructure)
{
	Ray ray = originalRay;
	while (voxelStructure->isRayInStructure(ray))
	{
		float nextX = ray.getDirection().getX() > 0.0f ? ceilf(ray.getOrigin().getX()) + EPSILON : floorf(ray.getOrigin().getX()) - EPSILON;
		float nextY = ray.getDirection().getY() > 0.0f ? ceilf(ray.getOrigin().getY()) + EPSILON : floorf(ray.getOrigin().getY()) - EPSILON;
		float nextZ = ray.getDirection().getZ() > 0.0f ? ceilf(ray.getOrigin().getZ()) + EPSILON : floorf(ray.getOrigin().getZ()) - EPSILON;
		//Calculate the t-values along the ray
		float tX = (nextX - ray.getOrigin().getX()) / ray.getDirection().getX();
		float tY = (nextY - ray.getOrigin().getY()) / ray.getDirection().getY();
		float tZ = (nextZ - ray.getOrigin().getZ()) / ray.getDirection().getZ();
		//Find the minimum t-value TODO add infinity consideration because of zero direction on ray
		float tMin = min(tX, min(tY, tZ));

		//Create the ray at the next position
		ray = Ray(ray.getOrigin() + (tMin + EPSILON) * ray.getDirection(), ray.getDirection());

		uint32_t x = static_cast<uint32_t>(ray.getOrigin().getX());
		uint32_t y = static_cast<uint32_t>(ray.getOrigin().getY());
		uint32_t z = static_cast<uint32_t>(ray.getOrigin().getZ());

		//Check if the voxel is in the map
		uint32_t voxelColor = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(x, y, z));
		if (voxelColor != EMPTY_KEY)
		{
			return voxelColor;
		}
	}

	//Return the background color of black
	return 0;
}

__device__ uint32_t rayMarchVoxelGridAxisJump(const Ray& originalRay, const VoxelStructure* voxelStructure)
{
	//Ray needs to be aligned to grid first in the longest direction -- then the loop can occur
	//Both -1.0f and 1.0f can be represented correctly so when orginally snapping to the grid an epsilon needs to be employed and will keep things the correct way

	uint32_t longestAxis;
	uint32_t middleAxis;
	uint32_t shortestAxis;

	Ray oldRay = originalRay.convertRayToLongestAxisDirection(originalRay, longestAxis, middleAxis, shortestAxis);
	Ray ray = oldRay;

	int32_t gridValues[3] = { static_cast<int32_t>(floorf(ray.getOrigin().getX())), static_cast<int32_t>(floorf(ray.getOrigin().getY())), static_cast<int32_t>(floorf(ray.getOrigin().getZ())) };
	int32_t longestAxisDiff = ray.getDirection()[longestAxis] < 0.0f ? -1 : 1;

	//Snap the longest direction vector axis to the grid first
	float t = ray.getDirection()[longestAxis] > 0.0f ?
		(gridValues[longestAxis] + EPSILON + 1 - ray.getOrigin()[longestAxis]) / ray.getDirection()[longestAxis] :
		(gridValues[longestAxis] - EPSILON - ray.getOrigin()[longestAxis]) / ray.getDirection()[longestAxis];
	ray = Ray(ray.getOrigin() + ray.getDirection() * t, ray.getDirection());
	int32_t middleAxisDiff = static_cast<int32_t>(floorf(ray.getOrigin()[middleAxis])) - gridValues[middleAxis];
	int32_t shortestAxisDiff = static_cast<int32_t>(floorf(ray.getOrigin()[shortestAxis])) - gridValues[shortestAxis];
	if (middleAxisDiff != 0 && shortestAxisDiff != 0)
	{
		float t1 = middleAxisDiff == -1 ?
			(std::floorf(oldRay.getOrigin()[middleAxis]) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis]
			: (std::ceilf(oldRay.getOrigin()[middleAxis]) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis];
		float shortestPosition = oldRay.getOrigin()[shortestAxis] + oldRay.getDirection()[shortestAxis] * t1;
		int32_t shorterDiff = static_cast<int32_t>(floorf(shortestPosition)) - gridValues[shortestAxis];
		if (shorterDiff != 0)
		{
			//apply shorter first
			gridValues[shortestAxis] += shortestAxisDiff;
			uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
			//apply longer second
			gridValues[middleAxis] += middleAxisDiff;
			colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
		}
		else
		{
			//apply longer first
			gridValues[middleAxis] += middleAxisDiff;
			uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
			//apply shorter second
			gridValues[shortestAxis] += shortestAxisDiff;
			colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
		}
	}
	else if (middleAxisDiff != 0)
	{
		gridValues[middleAxis] += middleAxisDiff;
		uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
		if (colorValue != EMPTY_KEY)
		{
			return colorValue;
		}
	}
	else if (shortestAxisDiff != 0)
	{
		gridValues[shortestAxis] += shortestAxisDiff;
		uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
		if (colorValue != EMPTY_KEY)
		{
			return colorValue;
		}
	}
	gridValues[longestAxis] += longestAxisDiff;
	uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
	if (colorValue != EMPTY_KEY)
	{
		return colorValue;
	}

	while (voxelStructure->isRayInStructure(ray))
	{
		Ray oldRay = ray;
		ray = Ray(ray.getOrigin() + ray.getDirection(), ray.getDirection());
		int32_t middleAxisDiff = static_cast<int32_t>(floorf(ray.getOrigin()[middleAxis])) - gridValues[middleAxis];
		int32_t shortestAxisDiff = static_cast<int32_t>(floorf(ray.getOrigin()[shortestAxis])) - gridValues[shortestAxis];
		if (middleAxisDiff != 0 && shortestAxisDiff != 0)
		{
			float t1 = middleAxisDiff == -1 ?
				(std::floorf(oldRay.getOrigin()[middleAxis]) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis]
				: (std::ceilf(oldRay.getOrigin()[middleAxis]) - oldRay.getOrigin()[middleAxis]) / oldRay.getDirection()[middleAxis];
			float shortestPosition = oldRay.getOrigin()[shortestAxis] + oldRay.getDirection()[shortestAxis] * t1;
			int32_t shorterDiff = static_cast<int32_t>(floorf(shortestPosition)) - gridValues[shortestAxis];
			if (shorterDiff != 0)
			{
				//apply shorter first
				gridValues[shortestAxis] += shortestAxisDiff;
				uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
				if (colorValue != EMPTY_KEY)
				{
					return colorValue;
				}
				//apply longer second
				gridValues[middleAxis] += middleAxisDiff;
				colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
				if (colorValue != EMPTY_KEY)
				{
					return colorValue;
				}
			}
			else
			{
				//apply longer first
				gridValues[middleAxis] += middleAxisDiff;
				uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
				if (colorValue != EMPTY_KEY)
				{
					return colorValue;
				}
				//apply shorter second
				gridValues[shortestAxis] += shortestAxisDiff;
				colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
				if (colorValue != EMPTY_KEY)
				{
					return colorValue;
				}
			}
		}
		else if (middleAxisDiff != 0)
		{
			gridValues[middleAxis] += middleAxisDiff;
			uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
		}
		else if (shortestAxisDiff != 0)
		{
			gridValues[shortestAxis] += shortestAxisDiff;
			uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
			if (colorValue != EMPTY_KEY)
			{
				return colorValue;
			}
		}
		gridValues[longestAxis] += longestAxisDiff;
		uint32_t colorValue = voxelStructure->hashTable->lookupValueForKey(generate3DInteger(gridValues[0], gridValues[1], gridValues[2]));
		if (colorValue != EMPTY_KEY)
		{
			return colorValue;
		}
	}
	return 0;
}

__global__ void rayMarchScene(uint32_t imgWidth, uint32_t imgHeight, Camera* camera, VoxelStructure* voxelStructure, uint8_t* framebuffer)
{
	uint32_t xPixel = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yPixel = threadIdx.y + blockIdx.y * blockDim.y;
	if (xPixel >= imgWidth || yPixel >= imgHeight) return;

	//Calculate normalized camera space coordinates (between 0 and 1) of the center of the pixel
	float uCamera = (xPixel + 0.5f) / static_cast<float>(imgWidth);
	//Since framebuffer coordinates have posative y going down and 3D position has a positive y going up. The imgHeight needs to be subracted to make up for this difference
	float vCamera = (imgHeight - yPixel + 0.5f) / static_cast<float>(imgHeight);

	//Calculate the ray cast by the camera with the camera space coordiantes for the pixel
	Ray ray = camera->generateRay(uCamera, vCamera);

	//Create the local ray which is in the coordinate domain of the voxel structure it will be traversing
	Ray localRay = ray.convertRayToLocalSpace(voxelStructure->translationVector, voxelStructure->scale);

	//Raymarch the voxel grid and get a color back
	uint32_t color = rayMarchVoxelGridAxisJump(localRay, voxelStructure);
	
	//Set the framebuffer location for that pixel to the returned color
	uint32_t pixelIndex = yPixel * imgWidth * 3 + xPixel * 3;
	framebuffer[pixelIndex] = color >> 16;
	framebuffer[pixelIndex + 1] = (color >> 8) & 0xFF;
	framebuffer[pixelIndex + 2] = color & 0xFF;
}