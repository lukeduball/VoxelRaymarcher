#pragma once

#include <unordered_map>

#include "VoxelSceneCPU.cuh"

class VoxelCube
{
public:
	static void generateVoxelCube(VoxelSceneCPU& voxelScene, int32_t xPos, int32_t yPos, int32_t zPos, int32_t halfWidth)
	{
		//Generate the voxels for the z axis faces
		for (int32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
		{
			for (int32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
			{
				voxelScene.insertVoxel(x, y, zPos + halfWidth, voxelfunc::generateRGBColor(255, 0, 0));
				voxelScene.insertVoxel(x, y, zPos - halfWidth, voxelfunc::generateRGBColor(255, 0, 0));
			}
		}
		//Generate the voxels for the x axis faces
		for (int32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
		{
			for (int32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
			{
				voxelScene.insertVoxel(xPos - halfWidth, y, z, voxelfunc::generateRGBColor(0, 255, 0));
				voxelScene.insertVoxel(xPos + halfWidth, y, z, voxelfunc::generateRGBColor(0, 255, 0));
			}
		}
		//Generate the voxels for the y axis faces
		for (int32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
		{
			for (int32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
			{
				voxelScene.insertVoxel(x, yPos - halfWidth, z, voxelfunc::generateRGBColor(0, 0, 255));
				voxelScene.insertVoxel(x, yPos + halfWidth, z, voxelfunc::generateRGBColor(0, 0, 255));
			}
		}
	}
};