#include <cuda_runtime.h>

#include "../geometry/VoxelCube.cuh"
#include "../geometry/VoxelFunctions.cuh"
#include "../geometry/VoxelSphere.cuh"

#include "../renderer/Renderer.cuh"
#include "../renderer/OptimizedFunctions.cuh"
#include "../renderer/VoxelStructure.cuh"
#include "../renderer/images/ImageWriter.h"
#include "../renderer/camera/Camera.cuh"

#include "../storage/CuckooHashTable.cuh"
#include "../storage/VoxelClusterStore.cuh"

#include <string>

#define DEVICE_ID 0

void setupConstantValues()
{
	Vector3 hostLightDirection = makeUnitVector(Vector3(1.0f, 1.0f, 1.0f));
	cudaMemcpyToSymbol(LIGHT_DIRECTION, &hostLightDirection, sizeof(Vector3));

	Vector3 hostLightColor = Vector3(1.0f, 1.0f, 1.0f);
	cudaMemcpyToSymbol(LIGHT_COLOR, &hostLightColor, sizeof(Vector3));

	Vector3 hostLightPosition = Vector3(10.0f, 10.0f, -10.0f);
	cudaMemcpyToSymbol(LIGHT_POSITION, &hostLightPosition, sizeof(Vector3));

	bool hostUsePointLight = true;
	cudaMemcpyToSymbol(USE_POINT_LIGHT, &hostUsePointLight, sizeof(bool));
}

int main(int argc, char* argv[])
{
	int32_t voxelLookupFunctionID = 0;
	int32_t rayMarchFunctionID = 0;
	bool useOptimizedFunctions = false;

	//If there are command line args, setup the commands
	if (argc == 3)
	{
		if (std::strcmp(argv[1], "hashtable") == 0)
		{
			voxelLookupFunctionID = 1;
			std::cout << "Storage Type: Cuckoo Hash Table" << std::endl;
		}
		else
		{
			std::cout << "Storage Type: Voxel Cluster Storage" << std::endl;
		}
		if (std::strcmp(argv[2], "original") == 0)
		{
			rayMarchFunctionID = 1;
			std::cout << "Raymarch Algorithm: Original" << std::endl;
		}
		else
		{
			std::cout << "Raymarch Algorithm: Jump Axis" << std::endl;
		}
	}
	else if (argc == 4)
	{
		useOptimizedFunctions = true;
		if (std::strcmp(argv[2], "hashtable") == 0)
		{
			voxelLookupFunctionID = 1;
			std::cout << "Storage Type: Cuckoo Hash Table" << std::endl;
		}
		else
		{
			std::cout << "Storage Type: Voxel Cluster Store" << std::endl;
		}
		if (std::strcmp(argv[3], "original") == 0)
		{
			rayMarchFunctionID = 1;
			std::cout << "Raymarch Algorithm: Original" << std::endl;
		}
		else
		{
			std::cout << "Raymarch Algorithm: Jump Axis" << std::endl;
		}
	}
	else
	{
		std::cout << "Using Default Arguments!" << std::endl;
	}

	int devCount;
	cudaGetDeviceCount(&devCount);
	cudaDeviceProp devProp;
	printf("Device Count: %d\n", devCount);
	if (devCount) 
	{
		cudaSetDevice(DEVICE_ID);
		cudaGetDeviceProperties(&devProp, DEVICE_ID);
	}
	printf("Device: %s\n", devProp.name);

	setupConstantValues();

	uint32_t width = 1920;
	uint32_t height = 1080;
	float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

	Camera camera = Camera(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, -1.0f), Vector3(0.0f, 1.0f, 0.0f), 90.0f, aspectRatio);
	//copy the created camera to the GPU
	Camera* deviceCamera;
	cudaMalloc(&deviceCamera, sizeof(Camera));
	cudaMemcpy(deviceCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	uint8_t* deviceFramebuffer;
	cudaMalloc(&deviceFramebuffer, sizeof(uint8_t) * width * height * 3);

	std::unordered_map<uint32_t, uint32_t> voxelMap;
	//VoxelCube::generateVoxelCube(voxelMap, 512, 512, 512, 50);
	VoxelSphere::generateVoxelSphere(voxelMap, BLOCK_SIZE / 2, BLOCK_SIZE / 2, BLOCK_SIZE / 2, BLOCK_SIZE / 6);

	//Create the GPU handles for both storage types
	CuckooHashTable* deviceHashTable = nullptr;
	VoxelClusterStore* deviceVoxelClusterStore = nullptr;

	//Case for the Voxel Cluster Store being the storage type
	if (voxelLookupFunctionID == 0)
	{
		//Generate the voxel cluster store on the CPU
		VoxelClusterStore voxelClusterStore = VoxelClusterStore(voxelMap);

		//Move the voxel cluster store to the GPU
		cudaMalloc(&deviceVoxelClusterStore, sizeof(VoxelClusterStore));
		cudaMemcpy(deviceVoxelClusterStore, &voxelClusterStore, sizeof(VoxelClusterStore), cudaMemcpyHostToDevice);
	}
	//Case for the Cuckoo Hash table being the storage type
	else
	{
		//Generate the cuckoo hash table on the CPU
		CuckooHashTable voxelHashTable = CuckooHashTable(voxelMap);

		//Move the hash table to the GPU
		cudaMalloc(&deviceHashTable, sizeof(CuckooHashTable));
		cudaMemcpy(deviceHashTable, &voxelHashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);
	}
	
	VoxelStructure voxelStructure = VoxelStructure(Vector3(-((float)BLOCK_SIZE / 2), -((float)BLOCK_SIZE / 2), -(float)BLOCK_SIZE), BLOCK_SIZE);

	//Copy the voxel structure to the GPU
	VoxelStructure* deviceVoxelStructure;
	cudaMalloc(&deviceVoxelStructure, sizeof(VoxelStructure));
	cudaMemcpy(deviceVoxelStructure, &voxelStructure, sizeof(VoxelStructure), cudaMemcpyHostToDevice);

	uint32_t numThreads = 8;
	dim3 blocks(width / numThreads + 1, height / numThreads + 1);
	dim3 threads(numThreads, numThreads);

	if (!useOptimizedFunctions)
	{
		if (rayMarchFunctionID == 0)
		{
			rayMarchSceneJumpAxis << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore, deviceHashTable);
		}
		else if (rayMarchFunctionID == 1)
		{
			rayMarchSceneOriginal << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore, deviceHashTable);
		}
	}
	else
	{
		//Jump Axis with VCS
		if (rayMarchFunctionID == 0 && voxelLookupFunctionID == 0)
		{
			//rayMarchSceneJumpAxisVCS << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore);
		}
		//Original with VCS
		else if (rayMarchFunctionID == 1 && voxelLookupFunctionID == 0)
		{
			rayMarchSceneOriginalVCS << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore);
		}
		//Jump Axis with Cuckoo Hash Table
		else if (rayMarchFunctionID == 0 && voxelLookupFunctionID == 1)
		{
			rayMarchSceneJumpAxisHashTable << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceHashTable);
		}
		//Original with Cuckoo Hash Table
		else
		{
			rayMarchSceneOriginalHashTable << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceHashTable);
		}
	}
	cudaError_t err = cudaPeekAtLastError();
	std::cout << cudaGetErrorString(err) << std::endl;

	cudaDeviceSynchronize();


	uint8_t* hostFramebuffer = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
	cudaMemcpy(hostFramebuffer, deviceFramebuffer, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);
	ImageWriter imgWriter = ImageWriter();
	imgWriter.writeImage("output.png", hostFramebuffer, width, height, 3);

	free(hostFramebuffer);
	cudaFree(deviceFramebuffer);
	cudaFree(deviceCamera);

	cudaFree(deviceVoxelStructure);

	if(deviceHashTable)
		cudaFree(deviceHashTable);
	if(deviceVoxelClusterStore)
		cudaFree(deviceVoxelClusterStore);
		
	return EXIT_SUCCESS;
}