#pragma once

#include <unordered_map>

#include "VoxelFunctions.cuh"

class VoxelCube
{
public:
	static void generateVoxelCube(std::unordered_map<uint32_t, uint32_t>& voxelTable, uint32_t xPos, uint32_t yPos, uint32_t zPos, uint32_t halfWidth)
	{
		//Generate the voxels for the z axis faces
		for (uint32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
		{
			for (uint32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
			{
				voxelTable[voxelfunc::generate3DPoint(x, y, zPos + halfWidth)] = voxelfunc::generateRGBColor(255, 0, 0);
				voxelTable[voxelfunc::generate3DPoint(x, y, zPos - halfWidth)] = voxelfunc::generateRGBColor(255, 0, 0);
			}
		}
		//Generate the voxels for the x axis faces
		for (uint32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
		{
			for (uint32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
			{
				voxelTable[voxelfunc::generate3DPoint(xPos - halfWidth, y, z)] = voxelfunc::generateRGBColor(0, 255, 0);
				voxelTable[voxelfunc::generate3DPoint(xPos + halfWidth, y, z)] = voxelfunc::generateRGBColor(0, 255, 0);
			}
		}
		//Generate the voxels for the y axis faces
		for (uint32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
		{
			for (uint32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
			{
				voxelTable[voxelfunc::generate3DPoint(x, yPos - halfWidth, z)] = voxelfunc::generateRGBColor(0, 255, 0);
				voxelTable[voxelfunc::generate3DPoint(x, yPos + halfWidth, z)] = voxelfunc::generateRGBColor(0, 255, 0);
			}
		}
	}
};