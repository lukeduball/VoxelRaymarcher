﻿#pragma once

#include "../geometry/VoxelFunctions.cuh"
#include "../math/Random.cuh"

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <unordered_map>

const uint32_t PRIME_NUMBER_TABLE_SIZE = 14;
const uint32_t PRIME_NUMBER_TABLE[] =
{
	668265261, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917, 25165843, 50331653
};

class CuckooHashTable
{
public:
	__host__ CuckooHashTable(const std::unordered_map<uint32_t, uint32_t>& existingTable) : numElements(static_cast<uint32_t>(existingTable.size())), 
		offset(0), primeConstant(PRIME_NUMBER_TABLE[0])
	{
		numElements = numElements * 1.25;
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

private:
	//Stores the GPU key/value pairs for the Cuckoo Hash Table
	uint32_t* deviceKey1Bucket;
	uint32_t* deviceValue1Bucket;
	uint32_t* deviceKey2Bucket;
	uint32_t* deviceValue2Bucket;
	//The number of elements in the hash table
	uint32_t numElements;

	uint32_t offset;
	uint32_t primeConstant;

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
		bool shouldRehash = false;

		do
		{
			shouldRehash = false;
			for (const auto& pair : existingTable)
			{
				uint32_t code = pair.first;
				uint32_t value = pair.second;

				//Determines which hash bucket to search in
				uint32_t bucket = 0;
				uint32_t maxIterations = 300000;
				uint32_t iterations = 0;
				while (true)
				{
					//A cycle has been detected and we need to choose new hash functions
					if (iterations >= maxIterations)
					{
						primeConstant = PRIME_NUMBER_TABLE[Random::getRandomInt(0, 14)];
						offset = Random::getRandomInt(0, 25);
						shouldRehash = true;
						//Fill all the keys in the array with the EMPTY_KEY
						std::fill_n(hostKey1Bucket, numElements, EMPTY_KEY);
						std::fill_n(hostKey2Bucket, numElements, EMPTY_KEY);
						std::fill_n(hostValue1Bucket, numElements, 0);
						std::fill_n(hostValue2Bucket, numElements, 0);
						break;
					}
					if (bucket == 0)
					{
						uint32_t key = (hashFunc1(code) % numElements) % numElements;
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
						uint32_t key = (hashFunc2(code) % numElements) % numElements;
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
					iterations++;
				}
			}
		} while (shouldRehash);
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
		return key + offset;
	}

	//Hash function needs to be visible to both the CPU(building hash table) and GPU(lookup functions)
	__host__ __device__ int hashFunc2(int key) const
	{
		unsigned int c2 = primeConstant; // a prime or an odd constant
		key = (key ^ 61) ^ (key >> 16);
		key = key + (key << 3);
		key = key ^ (key >> 4);
		key = key * c2;
		key = key ^ (key >> 15);
		return key;
	}
};