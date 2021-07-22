#include <cuda_runtime.h>

#include "../geometry/VoxelCube.cuh"
#include "../geometry/VoxelSphere.cuh"

#include "../renderer/Renderer.cuh"
#include "../renderer/VoxelStructure.cuh"
#include "../renderer/images/ImageWriter.h"
#include "../renderer/camera/Camera.cuh"

#include "../cuckoohash/CuckooHashTable.cuh"
#include "../voxelclusterstore/VoxelClusterStore.cuh"

#include <string>

#define DEVICE_ID 0

__global__ void testLookupFunc(uint32_t* lookupValues, uint32_t numLookupKeys, CuckooHashTable* hashTable)
{
	for (uint32_t i = threadIdx.x; i < numLookupKeys; i += blockDim.x)
	{
		uint32_t result = hashTable->lookupValueForKey(lookupValues[i]);
		printf("Index:%d Key:%d -> Value:%d\n", i, lookupValues[i], result);
	}
}

int main(int argc, char* argv[])
{
	int32_t voxelLookupFunctionID = 0;
	int32_t rayMarchFunctionID = 0;

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
	VoxelSphere::generateVoxelSphere(voxelMap, 32, 32, 32, 10);

	//Create the GPU handles for both storage types
	CuckooHashTable* deviceVoxelHashTable = nullptr;
	VoxelClusterStore* deviceVoxelClusterStore = nullptr;

	VoxelStructure voxelStructure;

	//Case for the Voxel Cluster Store being the storage type
	if (voxelLookupFunctionID == 0)
	{
		//Generate the voxel cluster store on the CPU
		VoxelClusterStore voxelClusterStore = VoxelClusterStore(voxelMap);

		//Move the voxel cluster store to the GPU
		cudaMalloc(&deviceVoxelClusterStore, sizeof(VoxelClusterStore));
		cudaMemcpy(deviceVoxelClusterStore, &voxelClusterStore, sizeof(VoxelClusterStore), cudaMemcpyHostToDevice);

		voxelStructure = VoxelStructure(deviceVoxelClusterStore, Vector3(-32.0f, -32.0f, -64.0f), 64);
	}
	//Case for the Cuckoo Hash table being the storage type
	else
	{
		//Generate the cuckoo hash table on the CPU
		CuckooHashTable voxelHashTable = CuckooHashTable(voxelMap);

		//Move the hash table to the GPU
		cudaMalloc(&deviceVoxelHashTable, sizeof(CuckooHashTable));
		cudaMemcpy(deviceVoxelHashTable, &voxelHashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);

		voxelStructure = VoxelStructure(deviceVoxelHashTable, Vector3(-32.0f, -32.0f, -64.0f), 64);
	}
	

	//Copy the voxel structure to the GPU
	VoxelStructure* deviceVoxelStructure;
	cudaMalloc(&deviceVoxelStructure, sizeof(VoxelStructure));
	cudaMemcpy(deviceVoxelStructure, &voxelStructure, sizeof(VoxelStructure), cudaMemcpyHostToDevice);

	uint32_t numThreads = 8;
	dim3 blocks(width / numThreads + 1, height / numThreads + 1);
	dim3 threads(numThreads, numThreads);
	rayMarchScene <<<blocks, threads>>> (width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer, rayMarchFunctionID, voxelLookupFunctionID);

	cudaDeviceSynchronize();


	uint8_t* hostFramebuffer = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
	cudaMemcpy(hostFramebuffer, deviceFramebuffer, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);
	ImageWriter imgWriter = ImageWriter();
	imgWriter.writeImage("output.png", hostFramebuffer, width, height, 3);

	free(hostFramebuffer);
	cudaFree(deviceFramebuffer);
	cudaFree(deviceCamera);

	cudaFree(deviceVoxelStructure);

	if(deviceVoxelHashTable)
		cudaFree(deviceVoxelHashTable);
	if (deviceVoxelClusterStore)
		cudaFree(deviceVoxelClusterStore);
		
	return EXIT_SUCCESS;
}