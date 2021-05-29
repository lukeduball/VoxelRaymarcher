#include <cuda_runtime.h>

#include <time.h>

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

	const uint32_t size = 2048;
	uint32_t* hostKeys = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
	uint32_t* hostValues = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));

	for (uint32_t i = 0; i < size; i++)
	{
		int key = generate3DInteger(rand() % 1024, rand() % 1024, rand() % 1024);
		hostKeys[i] = key;
		hostValues[i] = key;
	}

	CuckooHashTable hashTable = CuckooHashTable(hostKeys, hostValues, size);

	//Move the hash table handle to the GPU
	CuckooHashTable* deviceHashTable;
	cudaMalloc(&deviceHashTable, sizeof(CuckooHashTable));
	cudaMemcpy(deviceHashTable, &hashTable, sizeof(CuckooHashTable), cudaMemcpyHostToDevice);

	uint32_t* hostLookupKeys, * deviceLookupKeys;
	const uint32_t numLookupKeys = 128;
	hostLookupKeys = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numLookupKeys));
	cudaMalloc(&deviceLookupKeys, sizeof(uint32_t) * numLookupKeys);

	for (uint32_t i = 0; i < numLookupKeys; i++)
	{
		hostLookupKeys[i] = hostKeys[rand() % 2048];
	}

	cudaMemcpy(deviceLookupKeys, hostLookupKeys, numLookupKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);

	testLookupFunc<<<(numLookupKeys + 255) / 256, 256>>> (deviceLookupKeys, numLookupKeys, deviceHashTable);

	cudaDeviceSynchronize();

	free(hostKeys);
	free(hostValues);
	free(hostLookupKeys);

	cudaFree(deviceLookupKeys);
	cudaFree(deviceHashTable);
		
	return EXIT_SUCCESS;
}