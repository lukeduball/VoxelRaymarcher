#include <cuda_runtime.h>

#include "../geometry/VoxelCube.cuh"
#include "../geometry/VoxelFile.cuh"
#include "../geometry/VoxelFunctions.cuh"
#include "../geometry/VoxelSceneCPU.cuh"
#include "../geometry/VoxelSphere.cuh"

#include "../math/Random.cuh"
#include "../memory/MemoryUtils.h"

#include "../renderer/Renderer.cuh"
#include "../renderer/images/ImageWriter.h"
#include "../renderer/camera/Camera.cuh"

#include "../storage/CuckooHashTable.cuh"
#include "../storage/VoxelClusterStore.cuh"

#include <chrono>
#include <string>

#define DEVICE_ID 0

//Sets up the constant values that will not change during a kernel run
void setupConstantValues()
{
	Vector3f hostLightDirection = makeUnitVector(Vector3f(1.0f, 1.0f, 1.0f));
	cudaMemcpyToSymbol(LIGHT_DIRECTION, &hostLightDirection, sizeof(Vector3f));

	Vector3f hostLightColor = Vector3f(1.0f, 1.0f, 1.0f);
	cudaMemcpyToSymbol(LIGHT_COLOR, &hostLightColor, sizeof(Vector3f));

	Vector3f hostLightPosition = Vector3f(10.0f, 10.0f, -10.0f);
	cudaMemcpyToSymbol(LIGHT_POSITION, &hostLightPosition, sizeof(Vector3f));

	bool hostUsePointLight = false;
	cudaMemcpyToSymbol(USE_POINT_LIGHT, &hostUsePointLight, sizeof(bool));

	bool hostUseShadows = true;
	cudaMemcpyToSymbol(USE_SHADOWS, &hostUseShadows, sizeof(bool));
}

void pickCudaDevice()
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
}

void populateVoxelScene(VoxelSceneCPU& voxelScene)
{
	//Read the voxel scene from the following file
	VoxelFile::readVoxelFile(voxelScene, "scene.vox");

	//Takes the CPU memory stored in standard containers and writes the data to the device
	voxelScene.generateVoxelScene();
}

void runRaymarchingKernel(uint32_t width, uint32_t height, Camera* deviceCameraPtr, VoxelSceneInfo* deviceSceneInfoPtr, uint8_t* deviceFramebufferPtr)
{
	//Sets up the number of threads and blocks that are run on the GPU
	uint32_t numThreads = 8;
	dim3 blocks(width / numThreads + 1, height / numThreads + 1);
	dim3 threads(numThreads, numThreads);

	//Find the starting time for the clock
	auto startTime = std::chrono::high_resolution_clock::now();

	rayMarchSceneOriginal << <blocks, threads >> > (width, height, deviceCameraPtr, deviceSceneInfoPtr, deviceFramebufferPtr);
	
	cudaError_t err = cudaPeekAtLastError();
	std::cout << cudaGetErrorString(err) << std::endl;

	//Wait until the ray marching kernel has completed on the GPU
	cudaDeviceSynchronize();

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
	std::cout << "Execution Time for Ray Marching Algorithm is: " << duration.count() << " microseconds" << std::endl;
}

void writeResultingImageToDisk(uint32_t width, uint32_t height, uint8_t* deviceFramebufferPtr)
{
	uint8_t* hostFramebuffer = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
	cudaMemcpy(hostFramebuffer, deviceFramebufferPtr, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);
	ImageWriter imgWriter = ImageWriter();
	imgWriter.writeImage("output.png", hostFramebuffer, width, height, 3);

	//Clean up the host framebuffer memory after writing to the disk
	free(hostFramebuffer);
}

int main(int argc, char* argv[])
{
	Random::initialize();

	//Setup the command line arguments
	if (argc <= 1)
	{
		std::cout << "You need to provide a voxel scale" << std::endl;
		return 1;
	}
	int32_t scale = std::stoi(argv[1]);

	pickCudaDevice();
	//Pass values that won't change during a kernel call to the GPU
	setupConstantValues();

	uint32_t width = 1920;
	uint32_t height = 1080;
	float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

	Camera camera = Camera(Vector3f(6.0f, 2.0f, 6.0f), Vector3f(0.0f, 0.0f, -1.0f), Vector3f(0.0f, 1.0f, 0.0f), 60.0f, aspectRatio);
	//copy the created camera to the GPU
	CudaDeviceMemoryJanitor<Camera> deviceCameraJanitor(&camera, "Camera Memory");

	//Allocate the memory on the device that will hold the resulting image
	CudaDeviceMemoryJanitor<uint8_t> deviceFramebufferJanitor(width * height * 3, "Framebuffer Memory");

	VoxelSceneCPU voxelScene;
	//Populate the passed Voxel Scene with voxel data
	populateVoxelScene(voxelScene);
	
	VoxelSceneInfo voxelSceneInfo = VoxelSceneInfo(voxelScene.deviceVoxelScene, voxelScene.getArrayDiameter(), voxelScene.getMinCoord(), Vector3f(0.0f, 0.0f, 0.0f), scale);
	CudaDeviceMemoryJanitor<VoxelSceneInfo> deviceVoxelSceneInfoJanitor(&voxelSceneInfo, "Voxel Scene Info Memory");

	//Run the raymarching kernel with the specified options and scene
	runRaymarchingKernel(width, height, deviceCameraJanitor.devicePtr, deviceVoxelSceneInfoJanitor.devicePtr, deviceFramebufferJanitor.devicePtr);

	//Get the resulting image from the device and output it to the disk
	writeResultingImageToDisk(width, height, deviceFramebufferJanitor.devicePtr);

	//Clean up the memory on the device from the voxel scene
	voxelScene.cleanupVoxelScene();
		
	return EXIT_SUCCESS;
}