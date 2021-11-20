#include "VoxelSceneCPU.cuh"

#include <fstream>
#include <string>

class VoxelFile
{
public:
	static void readVoxelFile(VoxelSceneCPU& voxelScene, const std::string& filename)
	{
		std::string line;
		std::ifstream voxelFile("resources/"+filename);
		while (voxelFile)
		{
			std::vector<std::string> lineEntries;
			size_t start = 0;
			size_t end = 0;
			std::getline(voxelFile, line);
			//Destruct all comma seperated values in the line into an array of elements
			while ((start = line.find_first_not_of(",", end)) != std::string::npos)
			{
				end = line.find_first_of(",", start);
				lineEntries.push_back(line.substr(start, end - start));
			}

			//Process the elements as x,y,z,color and place the voxels in the scene
			if (lineEntries.size() > 3)
			{
				int32_t x = std::stoi(lineEntries[0]);
				int32_t y = std::stoi(lineEntries[1]);
				int32_t z = std::stoi(lineEntries[2]);
				uint32_t color = static_cast<uint32_t>(std::stoi(lineEntries[3]));
				voxelScene.insertVoxel(x, y, z, color);
			}
		}
	}
};