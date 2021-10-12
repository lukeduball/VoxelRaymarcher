#pragma once

#include "../geometry/VoxelFunctions.cuh"

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <unordered_map>

class CuckooHashTable
{
public:
	__host__ CuckooHashTable(uint32_t* keys, uint32_t* values, uint32_t size) : numElements(size)
	{
		//Allocate the memory for the Hash Table on both the CPU and GPU
		uint32_t* hostKey1Bucket, * hostKey2Bucket, * hostValue1Bucket, * hostValue2Bucket;
		hostAndDeviceAlloc(&hostKey1Bucket, &deviceKey1Bucket, size);
		hostAndDeviceAlloc(&hostKey2Bucket, &deviceKey2Bucket, size);
		hostAndDeviceAlloc(&hostValue1Bucket, &deviceValue1Bucket, size);
		hostAndDeviceAlloc(&hostValue2Bucket, &deviceValue2Bucket, size);

		//Fill all the keys in the array with the EMPTY_KEY
		std::fill_n(hostKey1Bucket, size, EMPTY_KEY);
		std::fill_n(hostKey2Bucket, size, EMPTY_KEY);

		//Create the Cuckoo Hash Table
		createCuckooHashTable(keys, values, hostKey1Bucket, hostKey2Bucket, hostValue1Bucket, hostValue2Bucket, size);

		//Copy the Cuckoo Hash Table to the GPU
		cudaMemcpy(deviceKey1Bucket, hostKey1Bucket, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceKey2Bucket, hostKey2Bucket, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceValue1Bucket, hostValue1Bucket, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceValue2Bucket, hostValue2Bucket, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

		//Free all the host memory
		free(hostKey1Bucket);
		free(hostKey2Bucket);
		free(hostValue1Bucket);
		free(hostValue2Bucket);
	}

	__host__ CuckooHashTable(const std::unordered_map<uint32_t, uint32_t>& existingTable) : numElements(static_cast<uint32_t>(existingTable.size()))
	{
		//Allocate the memory for the Hash Table on both the CPU and GPU
		uint32_t* hostKey1Bucket, * hostKey2Bucket, * hostValue1Bucket, * hostValue2Bucket;
		hostAndDeviceAlloc(&hostKey1Bucket, &deviceKey1Bucket, numElements);
		hostAndDeviceAlloc(&hostKey2Bucket, &deviceKey2Bucket, numElements);
		hostAndDeviceAlloc(&hostValue1Bucket, &deviceValue1Bucket, numElements);
		hostAndDeviceAlloc(&hostValue2Bucket, &deviceValue2Bucket, numElements);

		//Fill all the keys in the array with the EMPTY_KEY
		std::fill_n(hostKey1Bucket, numElements, EMPTY_KEY);
		std::fill_n(hostKey2Bucket, numElements, EMPTY_KEY);

		//Create the Cuckoo Hash Table
		createCuckooHashTable(existingTable, hostKey1Bucket, hostKey2Bucket, hostValue1Bucket, hostValue2Bucket);

		//Copy the Cuckoo Hash Table to the GPU
		cudaMemcpy(deviceKey1Bucket, hostKey1Bucket, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceKey2Bucket, hostKey2Bucket, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceValue1Bucket, hostValue1Bucket, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceValue2Bucket, hostValue2Bucket, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);

		//Free all the host memory
		free(hostKey1Bucket);
		free(hostKey2Bucket);
		free(hostValue1Bucket);
		free(hostValue2Bucket);
	}

	__host__ ~CuckooHashTable()
	{
		cudaFree(deviceKey1Bucket);
		cudaFree(deviceKey2Bucket);
		cudaFree(deviceValue1Bucket);
		cudaFree(deviceValue2Bucket);
	}

	__device__ uint32_t lookupVoxel(int32_t x, int32_t y, int32_t z) const
	{
		uint32_t code = voxelfunc::generate3DPoint(x, y, z);
		uint32_t key1 = (hashFunc1(code) % numElements + numElements) % numElements;
		if (deviceKey1Bucket[key1] == code)
		{
			return deviceValue1Bucket[key1];
		}
		else
		{
			uint32_t key2 = (hashFunc2(code) % numElements + numElements) % numElements;
			if (deviceKey2Bucket[key2] == code)
			{
				return deviceValue2Bucket[key2];
			}
		}
		return EMPTY_KEY;
	}

	__host__ void printHashTableGPUValues()
	{
		uint32_t* hostMemory = static_cast<uint32_t*>(malloc(numElements * sizeof(uint32_t)));
		cudaMemcpy(hostMemory, deviceKey1Bucket, numElements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i < numElements; i++)
		{
			if (hostMemory[i] != EMPTY_KEY)
			{
				std::cout << hostMemory[i] << " ";
			}
		}
		cudaMemcpy(hostMemory, deviceKey2Bucket, numElements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i < numElements; i++)
		{
			if (hostMemory[i] != EMPTY_KEY)
			{
				std::cout << hostMemory[i] << " ";
			}
		}
		std::cout << "\n\n";
		cudaMemcpy(hostMemory, deviceValue1Bucket, numElements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i < numElements; i++)
		{
			if (hostMemory[i] != EMPTY_KEY)
			{
				std::cout << hostMemory[i] << " ";
			}
		}
		cudaMemcpy(hostMemory, deviceValue2Bucket, numElements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i < numElements; i++)
		{
			if (hostMemory[i] != EMPTY_KEY)
			{
				std::cout << hostMemory[i] << " ";
			}
		}
		std::cout << "\n";
		free(hostMemory);
	}

private:
	//Stores the GPU key/value pairs for the Cuckoo Hash Table
	uint32_t* deviceKey1Bucket;
	uint32_t* deviceValue1Bucket;
	uint32_t* deviceKey2Bucket;
	uint32_t* deviceValue2Bucket;
	//The number of elements in the hash table
	uint32_t numElements;

	//Allocates memory for the hash table on both the CPU and GPU
	__host__ void hostAndDeviceAlloc(uint32_t** hostVar, uint32_t** deviceVar, uint32_t size)
	{
		*hostVar = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
		cudaMalloc(deviceVar, sizeof(uint32_t) * size);
	}

	__host__ void createCuckooHashTable(const std::unordered_map<uint32_t, uint32_t>& existingTable, 
		uint32_t* hostKey1Bucket, uint32_t* hostKey2Bucket,
		uint32_t* hostValue1Bucket, uint32_t* hostValue2Bucket)
	{
		uint32_t size = static_cast<uint32_t>(existingTable.size());

		for (const auto& pair : existingTable)
		{
			uint32_t code = pair.first;
			uint32_t value = pair.second;
		
			//Determines which hash bucket to search in
			uint32_t bucket = 0;
			while (true)
			{
				if (bucket == 0)
				{
					uint32_t key = (hashFunc1(code) % size) % size;
					//If the key is empty insert the key/value pair
					if (hostKey1Bucket[key] == EMPTY_KEY)
					{
						hostKey1Bucket[key] = code;
						hostValue1Bucket[key] = value;
						break;
					}
					//If the key is not empty insert the key/value pair and look for new location for existing key/value pair
					else
					{
						uint32_t tempCode = hostKey1Bucket[key];
						uint32_t tempValue = hostValue1Bucket[key];
						hostKey1Bucket[key] = code;
						hostValue1Bucket[key] = value;
						code = tempCode;
						value = tempValue;
						bucket = 1;
					}
				}
				else
				{
					uint32_t key = (hashFunc2(code) % size) % size;
					//If the key is empty insert the key/value pair
					if (hostKey2Bucket[key] == EMPTY_KEY)
					{
						hostKey2Bucket[key] = code;
						hostValue2Bucket[key] = value;
						break;
					}
					//If the key is not empty insert the key/value pair and look for new location for existing key/value pair
					else
					{
						uint32_t tempCode = hostKey2Bucket[key];
						uint32_t tempValue = hostValue2Bucket[key];
						hostKey2Bucket[key] = code;
						hostValue2Bucket[key] = value;
						code = tempCode;
						value = tempValue;
						bucket = 0;
					}
				}
			}
		}
	}

	__host__ void createCuckooHashTable(uint32_t* keys, uint32_t* values, 
		uint32_t* hostKey1Bucket, uint32_t* hostKey2Bucket, 
		uint32_t* hostValue1Bucket, uint32_t* hostValue2Bucket, 
		uint32_t size)
	{
		for (uint32_t i = 0; i < size; i++)
		{
			uint32_t code = keys[i];
			uint32_t value = values[i];

			//Determines which hash bucket to search in
			uint32_t bucket = 0;
			while (true)
			{
				if (bucket == 0)
				{
					uint32_t key = (hashFunc1(code) % size) % size;
					//If the key is empty insert the key/value pair
					if (hostKey1Bucket[key] == EMPTY_KEY)
					{
						hostKey1Bucket[key] = code;
						hostValue1Bucket[key] = value;
						break;
					}
					//If the key is not empty insert the key/value pair and look for new location for existing key/value pair
					else
					{
						uint32_t tempCode = hostKey1Bucket[key];
						uint32_t tempValue = hostValue1Bucket[key];
						hostKey1Bucket[key] = code;
						hostValue1Bucket[key] = value;
						code = tempCode;
						value = tempValue;
						bucket = 1;
					}
				}
				else
				{
					uint32_t key = (hashFunc2(code) % size) % size;
					//If the key is empty insert the key/value pair
					if (hostKey2Bucket[key] == EMPTY_KEY)
					{
						hostKey2Bucket[key] = code;
						hostValue2Bucket[key] = value;
						break;
					}
					//If the key is not empty insert the key/value pair and look for new location for existing key/value pair
					else
					{
						uint32_t tempCode = hostKey2Bucket[key];
						uint32_t tempValue = hostValue2Bucket[key];
						hostKey2Bucket[key] = code;
						hostValue2Bucket[key] = value;
						code = tempCode;
						value = tempValue;
						bucket = 0;
					}
				}
			}
		}
	}

	//Hash function needs to be visible to both the CPU(building hash table) and GPU(lookup functions)
	__host__ __device__ int hashFunc1(int key) const
	{
		key = (key + 0x7ed55d16) + (key << 12);
		key = (key ^ 0xc761c23c) ^ (key >> 19);
		key = (key + 0x165667b1) + (key << 5);
		key = (key + 0xd3a2646c) ^ (key << 9);
		key = (key + 0xfd7046c5) + (key << 3);
		key = (key ^ 0xb55a4f09) ^ (key >> 16);
		return key;
	}

	//Hash function needs to be visible to both the CPU(building hash table) and GPU(lookup functions)
	__host__ __device__ int hashFunc2(int key) const
	{
		unsigned int c2 = 0x27d4eb2d; // a prime or an odd constant
		key = (key ^ 61) ^ (key >> 16);
		key = key + (key << 3);
		key = key ^ (key >> 4);
		key = key * c2;
		key = key ^ (key >> 15);
		return key;
	}
};