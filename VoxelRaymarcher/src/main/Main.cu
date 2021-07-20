#include <cuda_runtime.h>

#include "../geometry/VoxelCube.cuh"
#include "../geometry/VoxelSphere.cuh"

#include "../renderer/Renderer.cuh"
#include "../renderer/VoxelStructure.cuh"
#include "../renderer/images/ImageWriter.h"
#include "../renderer/camera/Camera.cuh"

#include "../cuckoohash/CuckooHashTable.cuh"
#include "../voxelclusterstore/VoxelClusterStore.cuh"

#define DEVICE_ID 0

__global__ void testLookupFunc(uint32_t* lookupValues, uint32_t numLookupKeys, CuckooHashTable* hashTable)
{
	for (uint32_t i = threadIdx.x; i < numLookupKeys; i += blockDim.x)
	{
		uint32_t result = hashTable->lookupValueForKey(lookupValues[i]);
		printf("Index:%d Key:%d -> Value:%d\n", i, lookupValues[i], result);
	}
}

int main()
{
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

	ImageWriter imgWriter = ImageWriter();

	uint8_t* deviceFramebuffer;
	cudaMalloc(&deviceFramebuffer, sizeof(uint8_t) * width * height * 3);

	std::unordered_map<uint32_t, uint32_t> voxelMap;
	//VoxelCube::generateVoxelCube(voxelMap, 512, 512, 512, 50);
	VoxelSphere::generateVoxelSphere(voxelMap, 32, 32, 32, 10);
	VoxelClusterStore voxelClusterStore = VoxelClusterStore(voxelMap);
	CuckooHashTable voxelHashTable = CuckooHashTable(voxelMap);

	//Voxel Cluster Store's GPU handle
	VoxelClusterStore* deviceVoxelClusterStore;
	//Move the voxel cluster store to the GPU
	cudaMalloc(&deviceVoxelClusterStore, sizeof(VoxelClusterStore));
	cudaMemcpy(deviceVoxelClusterStore, &voxelClusterStore, sizeof(VoxelClusterStore), cudaMemcpyHostToDevice);

	//Hash table's GPU handle
	CuckooHashTable* deviceVoxelHashTable;
	//Move the hash table to the GPU
	cudaMalloc(&deviceVoxelHashTable, sizeof(CuckooHashTable));
	cudaMemcpy(deviceVoxelHashTable, &voxelHashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);

	VoxelStructure voxelStructure = VoxelStructure(deviceVoxelClusterStore, Vector3(-32.0f, -32.0f, -64.0f), 64);

	//Copy the voxel structure to the GPU
	VoxelStructure* deviceVoxelStructure;
	cudaMalloc(&deviceVoxelStructure, sizeof(VoxelStructure));
	cudaMemcpy(deviceVoxelStructure, &voxelStructure, sizeof(VoxelStructure), cudaMemcpyHostToDevice);

	uint32_t numThreads = 8;
	dim3 blocks(width / numThreads + 1, height / numThreads + 1);
	dim3 threads(numThreads, numThreads);
	rayMarchScene <<<blocks, threads>>>(width, height, deviceCamera, deviceVoxelStructure, deviceFramebuffer);

	cudaDeviceSynchronize();

	uint8_t* hostFramebuffer = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
	cudaMemcpy(hostFramebuffer, deviceFramebuffer, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);
	imgWriter.writeImage("output.png", hostFramebuffer, width, height, 3);

	free(hostFramebuffer);
	cudaFree(deviceFramebuffer);
	cudaFree(deviceCamera);

	cudaFree(deviceVoxelStructure);
	cudaFree(deviceVoxelHashTable);
		
	return EXIT_SUCCESS;
}