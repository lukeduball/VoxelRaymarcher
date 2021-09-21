#pragma once

#include <unordered_map>

#include "VoxelFunctions.cuh"

class VoxelSphere
{
public:
	static void generateVoxelSphere(std::unordered_map<uint32_t, uint32_t>& voxelTable, uint32_t xPos, uint32_t yPos, uint32_t zPos, uint32_t radius)
	{
		uint32_t xMin = xPos - radius;
		uint32_t xMax = xPos + radius;
		uint32_t yMin = yPos - radius;
		uint32_t yMax = yPos + radius;
		uint32_t zMin = zPos - radius;
		uint32_t zMax = zPos + radius;

		for (uint32_t x = xMin; x < xMax; x++)
		{
			for (uint32_t y = yMin; y < yMax; y++)
			{
				for (uint32_t z = zMin; z < zMax; z++)
				{
					uint32_t distanceSquared = (x - xPos) * (x - xPos) + (y - yPos) * (y - yPos) + (z - zPos) * (z - zPos);
					if (distanceSquared < radius * radius && distanceSquared >(radius - 1) * (radius - 1))
					{
						uint32_t red = 50 + (uint32_t)((x - xMin) * (200 / (xMax - xMin)));
						uint32_t green = 50 + (uint32_t)((y - xMin) * (200 / (yMax - yMin)));
						uint32_t blue = 50 + (uint32_t)((z - zMin) * (200 / (xMax - zMin)));
						voxelTable[voxelfunc::generate3DPoint(x, y, z)] = voxelfunc::generateRGBColor(std::min(red, 255u), std::min(green, 255u), std::min(blue, 255u));
					}
				}
			}
		}
	}

	static void generateCheckeredVoxelSphere(std::unordered_map<uint32_t, uint32_t>& voxelTable, uint32_t xPos, uint32_t yPos, uint32_t zPos, uint32_t radius)
	{
		uint32_t xMin = xPos - radius;
		uint32_t xMax = xPos + radius;
		uint32_t yMin = yPos - radius;
		uint32_t yMax = yPos + radius;
		uint32_t zMin = zPos - radius;
		uint32_t zMax = zPos + radius;

		for (uint32_t x = xMin; x < xMax; x+=2)
		{
			for (uint32_t y = yMin; y < yMax; y++)
			{
				for (uint32_t z = zMin; z < zMax; z++)
				{
					uint32_t distanceSquared = (x - xPos) * (x - xPos) + (y - yPos) * (y - yPos) + (z - zPos) * (z - zPos);
					if (distanceSquared < radius * radius && distanceSquared > (radius - 1) * (radius - 1))
					{
						uint32_t red = 50 + (uint32_t)((x - xMin) * (200 / (xMax - xMin)));
						uint32_t green = 50 + (uint32_t)((y - xMin) * (200 / (yMax - yMin)));
						uint32_t blue = 50 + (uint32_t)((z - zMin) * (200 / (xMax - zMin)));
						voxelTable[voxelfunc::generate3DPoint(x, y, z)] = voxelfunc::generateRGBColor(red, green, blue);
					}
				}
			}
		}
	}
};