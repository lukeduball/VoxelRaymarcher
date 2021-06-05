#include "VoxelCube.h"

#include "VoxelFunctions.h"

void VoxelCube::generateVoxelCube(std::unordered_map<uint32_t, uint32_t>& voxelTable, uint32_t xPos, uint32_t yPos, uint32_t zPos, uint32_t halfWidth)
{
	//Generate the voxels for the z axis faces
	for (uint32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
	{
		for (uint32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
		{
			voxelTable[VoxelFunctions::generate3DPoint(x, y, zPos + halfWidth)] = VoxelFunctions::generateRGBColor(255, 0, 0);
			voxelTable[VoxelFunctions::generate3DPoint(x, y, zPos - halfWidth)] = VoxelFunctions::generateRGBColor(255, 0, 0);
		}
	}
	//Generate the voxels for the x axis faces
	for (uint32_t y = yPos - halfWidth; y < yPos + halfWidth; y++)
	{
		for (uint32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
		{
			voxelTable[VoxelFunctions::generate3DPoint(xPos - halfWidth, y, z)] = VoxelFunctions::generateRGBColor(0, 255, 0);
			voxelTable[VoxelFunctions::generate3DPoint(xPos + halfWidth, y, z)] = VoxelFunctions::generateRGBColor(0, 255, 0);
		}
	}
	//Generate the voxels for the y axis faces
	for (uint32_t x = xPos - halfWidth; x < xPos + halfWidth; x++)
	{
		for (uint32_t z = zPos - halfWidth; z < zPos + halfWidth; z++)
		{
			voxelTable[VoxelFunctions::generate3DPoint(x, yPos - halfWidth, z)] = VoxelFunctions::generateRGBColor(0, 255, 0);
			voxelTable[VoxelFunctions::generate3DPoint(x, yPos + halfWidth, z)] = VoxelFunctions::generateRGBColor(0, 255, 0);
		}
	}
}