#include <cuda_runtime.h>

#include <time.h>

#include "../geometry/VoxelCube.h"
#include "../geometry/VoxelSphere.h"

#include "../cuckoohash/CuckooHashTable.cuh"

#define DEVICE_ID 0

int generate3DInteger(int x, int y, int z)
{
	return (x << 20) | (y << 10) | z;
}

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
	if (devCount) {
		cudaSetDevice(DEVICE_ID);
		cudaGetDeviceProperties(&devProp, DEVICE_ID);
	}
	printf("Device: %s\n", devProp.name);

	srand(time(NULL));

	std::unordered_map<uint32_t, uint32_t> voxelMap;
	//VoxelCube::generateVoxelCube(voxelMap, 512, 512, 512, 50);
	VoxelSphere::generateVoxelSphere(voxelMap, 512, 512, 512, 50);
	CuckooHashTable voxelHashTable = CuckooHashTable(voxelMap);

	//Hash table's GPU handle
	CuckooHashTable* deviceVoxelHashTable;
	//Move the hash table to the GPU
	cudaMalloc(&deviceVoxelHashTable, sizeof(CuckooHashTable));
	cudaMemcpy(deviceVoxelHashTable, &voxelHashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);

	uint32_t* hostLookupKeys, * deviceLookupKeys;
	const uint32_t numLookupKeys = 128;
	hostLookupKeys = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numLookupKeys));
	cudaMalloc(&deviceLookupKeys, sizeof(uint32_t) * numLookupKeys);

	uint32_t counter = 0;
	for (auto iterator = voxelMap.begin(); iterator != voxelMap.end(); ++iterator)
	{
		if (counter >= numLookupKeys)
			break;
		hostLookupKeys[counter] = iterator->first;
		counter++;
	}

	cudaMemcpy(deviceLookupKeys, hostLookupKeys, numLookupKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);

	testLookupFunc<<<(numLookupKeys + 255) / 256, 256>>> (deviceLookupKeys, numLookupKeys, deviceVoxelHashTable);

	cudaDeviceSynchronize();

	free(hostLookupKeys);

	cudaFree(deviceLookupKeys);
	cudaFree(deviceVoxelHashTable);
		
	return EXIT_SUCCESS;
}