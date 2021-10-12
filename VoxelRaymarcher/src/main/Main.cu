#include <cuda_runtime.h>

#include "../geometry/VoxelCube.cuh"
#include "../geometry/VoxelFunctions.cuh"
#include "../geometry/VoxelSceneCPU.cuh"
#include "../geometry/VoxelSphere.cuh"

#include "../renderer/Renderer.cuh"
//#include "../renderer/OptimizedFunctions.cuh"
#include "../renderer/images/ImageWriter.h"
#include "../renderer/camera/Camera.cuh"

#include "../storage/CuckooHashTable.cuh"
#include "../storage/VoxelClusterStore.cuh"

#include <string>

#define DEVICE_ID 0

void setupConstantValues()
{
	Vector3f hostLightDirection = makeUnitVector(Vector3f(-1.0f, 1.0f, 1.0f));
	cudaMemcpyToSymbol(LIGHT_DIRECTION, &hostLightDirection, sizeof(Vector3f));

	Vector3f hostLightColor = Vector3f(1.0f, 1.0f, 1.0f);
	cudaMemcpyToSymbol(LIGHT_COLOR, &hostLightColor, sizeof(Vector3f));

	Vector3f hostLightPosition = Vector3f(10.0f, 10.0f, -10.0f);
	cudaMemcpyToSymbol(LIGHT_POSITION, &hostLightPosition, sizeof(Vector3f));

	bool hostUsePointLight = false;
	cudaMemcpyToSymbol(USE_POINT_LIGHT, &hostUsePointLight, sizeof(bool));

	bool hostUseShadows = false;
	cudaMemcpyToSymbol(USE_SHADOWS, &hostUseShadows, sizeof(bool));
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

	Camera camera = Camera(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, -1.0f), Vector3f(0.0f, 1.0f, 0.0f), 90.0f, aspectRatio);
	//copy the created camera to the GPU
	Camera* deviceCamera;
	cudaMalloc(&deviceCamera, sizeof(Camera));
	cudaMemcpy(deviceCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	uint8_t* deviceFramebuffer;
	cudaMalloc(&deviceFramebuffer, sizeof(uint8_t) * width * height * 3);

	VoxelSceneCPU voxelScene;
	VoxelSphere::generateVoxelSphere(voxelScene, BLOCK_SIZE / 2, BLOCK_SIZE / 2, BLOCK_SIZE / 2, BLOCK_SIZE / 6);
	VoxelSphere::generateVoxelSphere(voxelScene, 42, 42, 45, 2);
	VoxelCube::generateVoxelCube(voxelScene, BLOCK_SIZE, BLOCK_SIZE / 2 + 5, BLOCK_SIZE / 2, BLOCK_SIZE / 6);

	voxelScene.generateVoxelScene(StorageType(voxelLookupFunctionID));
	StorageStructure** deviceVoxelScene;
	cudaMalloc(&deviceVoxelScene, sizeof(StorageStructure*) * voxelScene.getArraySize());
	generateVoxelScene<<<1, 1>>>(deviceVoxelScene, voxelScene.deviceVoxelScene, voxelScene.getArraySize(), StorageType(voxelLookupFunctionID));

	cudaDeviceSynchronize();
	
	VoxelSceneInfo voxelSceneInfo = VoxelSceneInfo(Vector3f(-((float)BLOCK_SIZE / 2), -((float)BLOCK_SIZE / 2), -(float)BLOCK_SIZE));
	VoxelSceneInfo* deviceVoxelSceneInfo;
	cudaMalloc(&deviceVoxelSceneInfo, sizeof(VoxelSceneInfo));
	cudaMemcpy(deviceVoxelSceneInfo, &voxelSceneInfo, sizeof(VoxelSceneInfo), cudaMemcpyHostToDevice);

	uint32_t numThreads = 8;
	dim3 blocks(width / numThreads + 1, height / numThreads + 1);
	dim3 threads(numThreads, numThreads);

	if (!useOptimizedFunctions)
	{
		if (rayMarchFunctionID == 0)
		{
			rayMarchSceneJumpAxis << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelSceneInfo, deviceFramebuffer,
				deviceVoxelScene, voxelScene.getArraySize(), StorageType(voxelLookupFunctionID));
		}
		else if (rayMarchFunctionID == 1)
		{
			rayMarchSceneOriginal << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelSceneInfo, deviceFramebuffer,
				deviceVoxelScene, voxelScene.getArraySize(), StorageType(voxelLookupFunctionID));
		}
	}
	else
	{
		std::cout << "Optimized functions are currently disabled" << std::endl;
		//Jump Axis with VCS
		if (rayMarchFunctionID == 0 && voxelLookupFunctionID == 0)
		{
			//rayMarchSceneJumpAxisVCS << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore);
		}
		//Original with VCS
		else if (rayMarchFunctionID == 1 && voxelLookupFunctionID == 0)
		{
			//rayMarchSceneOriginalVCS << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceVoxelClusterStore);
		}
		//Jump Axis with Cuckoo Hash Table
		else if (rayMarchFunctionID == 0 && voxelLookupFunctionID == 1)
		{
			//rayMarchSceneJumpAxisHashTable << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceHashTable);
		}
		//Original with Cuckoo Hash Table
		else
		{
			//rayMarchSceneOriginalHashTable << <blocks, threads >> > (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, deviceHashTable);
		}
	}
	cudaError_t err = cudaPeekAtLastError();
	std::cout << cudaGetErrorString(err) << std::endl;

	cudaDeviceSynchronize();


	uint8_t* hostFramebuffer = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
	cudaMemcpy(hostFramebuffer, deviceFramebuffer, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);
	ImageWriter imgWriter = ImageWriter();
	imgWriter.writeImage("output.png", hostFramebuffer, width, height, 3);

	cudaFree(deviceVoxelScene);

	free(hostFramebuffer);
	cudaFree(deviceFramebuffer);
	cudaFree(deviceCamera);
	cudaFree(deviceVoxelSceneInfo);

	voxelScene.cleanupVoxelScene();
		
	return EXIT_SUCCESS;
}