#include <cuda_runtime.h>

class Ray
{
public:
	__device__ Ray(float3 origin, float3 direction);

	__device__ float3 origin;
	__device__ float3 direction;
};