#pragma once

#include <cuda_runtime.h>
#include <iostream>

class CudaMemoryUtils
{
public:
	static void ManagedCudaFree(void* devicePtr, std::string memoryDescription)
	{
		std::cout << "Freeing CUDA Memory (Manual): " << memoryDescription << std::endl;
		cudaFree(devicePtr);
	}
};

template<typename T>
class CudaDeviceMemoryJanitor
{
public:
	CudaDeviceMemoryJanitor(T* hostPtr);
	CudaDeviceMemoryJanitor(T* hostPtr, std::string memoryDescription);
	CudaDeviceMemoryJanitor(size_t dataSize);
	CudaDeviceMemoryJanitor(size_t dataSize, std::string memoryDescription);
	~CudaDeviceMemoryJanitor();

	T* devicePtr;
	std::string memoryDescriptor;
};

template<typename T>
inline CudaDeviceMemoryJanitor<T>::CudaDeviceMemoryJanitor(T* hostPtr) : memoryDescriptor("")
{
	cudaMalloc(&devicePtr, sizeof(T));
	cudaMemcpy(devicePtr, hostPtr, sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline CudaDeviceMemoryJanitor<T>::CudaDeviceMemoryJanitor(T* hostPtr, std::string memoryDescription) : CudaDeviceMemoryJanitor(hostPtr)
{
	memoryDescriptor = memoryDescription;
}

template<typename T>
inline CudaDeviceMemoryJanitor<T>::CudaDeviceMemoryJanitor(size_t dataSize)
{
	cudaMalloc(&devicePtr, sizeof(T) * dataSize);
}

template<typename T>
inline CudaDeviceMemoryJanitor<T>::CudaDeviceMemoryJanitor(size_t dataSize, std::string memoryDescription) : CudaDeviceMemoryJanitor(dataSize)
{
	memoryDescriptor = memoryDescription;
}

template<typename T>
inline CudaDeviceMemoryJanitor<T>::~CudaDeviceMemoryJanitor()
{
	std::cout << "Freeing CUDA Memory (Janitor): " << memoryDescriptor << std::endl;
	cudaFree(devicePtr);
}
