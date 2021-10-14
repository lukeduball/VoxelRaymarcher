# VoxelRaymarcher
## Description
---
This project implements a Voxel Raymarching Renderer. The scenes that are raymarched consist entirely of axis aligned voxels which are given a color value. Rays are then cast into the scene for each pixel and "marched" over the underlaying voxel structure in order to find if a ray intersects a voxel. If it does, lighting is applied to the color and its color value is written back to the pixel location. Using this method, the entire scene is rendered.

## Setup Instructions
Download the Cuda Toolkit from https://developer.nvidia.com/cuda-toolkit

Download and install Visual Studio 2019

1. Clone the repository into a new folder
2. Navigate to the VoxelRaymarcher folder and open the VoxelRaymarcher.sln with Visual Studio 2019
3. Build the software (all dependencies are included and the path to the Cuda Toolkit should will be populated after installing the Cuda Toolkit from the above link)

## Running the Raymarcher
After performing a software build the executables can be run in the following way.

1. Open the command prompt
2. Navigate to VoxelRaymarcher/x64/Debug/ or navigate to VoxelRaymarcher/x64/Release/ based on the configuration that it was built with
3. Run the program with the desired command
    - Original Raymarching Algorithm with a Hashtable `./VoxelRaymarcher.exe hashtable original`
    - Original Raymarching Algorithm with a Voxel Cluster Store `./VoxelRaymarcher.exe vcs original`
    - Longest Axis Raymarching Algorithm with a Hashtable `./VoxelRaymarcher.exe hashtable jumpaxis`
    - Longest Axis Raymarching Algorithm with a Voxel Cluster Store `./VoxelRaymarcher.exe vcs jumpaxis`
4. View the resulting image in the VoxelRaymarcher/x64/[Debug/Release]/ folder with the name `output.png`

## Raymarching Algorithms
---
Two different type of raymarching algorithms are implemented as methods to traverse the underlaying axis aligned grid which contains the voxels. The two methods are the original raymarching algorithm and the jump axis raymarching algorithm.
### Original Raymarching Algorithm
The original raymarching algorithm traverses the voxel grid by starting at the rays original location. The ray then checks all three axis using its direction vector to find the next closest intersection for the next voxel location the ray will intersect. Using this voxel location, a lookup is performed using the storage structure to see if a voxel (color value) exists at that location. If it does, lighting is applied and that color value is returned. If not, the algorithm looks for the next voxel location until it leaves the voxel grid. If it leaves the grid it will just return the background color.

### Jump Axis Raymarching Algorithm
The jump axis raymarching algorithm takes advantage of the idea that the axis with the largest direction component for its direction vector will be the axis that a voxel location is found at the most. Therefore, the ray is scaled along the longest direction to have a value of 1 in that direction. In this way, the ray can "jump" over that entire axis to perform less iterations. We still can not forget about the other axes, so a way to tell if another axis has been intersected first is to keep track of the integer voxel locations (for all axis). When an axis jump is performed, check the new integer voxel locations of the new ray. Find the difference between the two integer voxel locations for each axes. If a component has a difference that is not 0, the voxel location for that axis needs to be checked for a voxel first. There are 4 cases that can occur for this raymarching algorithm while the longest axis check will always occur for each iteration.
1. Shorest Axis only needs to be checked
2. Middle Axis only needs to be checked
3. Shortest Axis needs to be checked first, the Middle Axis needs to be checked second
4. Middle Axis needs to be checked first, the Shortest Axis needs to be checked second

## Storage Structures
---
The storage structure is the way the voxels are stored and retrieved from memory. Two different methods have been employed for use as a storage structure.

### Cuckoo Hash Table
The Cuckoo Hash Table is a simple storage structure that takes advantage of two level hash table. The advantage of this type of algorithm is that the lookup time is constant O(1). It also stores memory pretty sparsly but does require double the amount of memory allocated per voxel. The biggest drawback of this method is that voxels are randomly distributed in memory (voxels close to each other in space, may not be close to each other in memory). This causes poor memory access patterns and a lot of cache misses, slowing down rendering time.

### Voxel Cluster Store (VCS)
Due to the poor memory access of the Cuckoo Hash Table, Voxel Cluster Stores were created as a storage structure. These storage structures work by taking the voxel grid for a certain block size and splitting it up into equal blocks of voxels. I.E. given a 64x64x64 block size of voxels, VCS can split up the voxels into blocks(clusters) of 8x8x8. The voxels are sorted and lookup is performed by a binary search for each cluster. This fixes the memory locality issue by making sure close voxels are stored together. It also is a more sparse way of storage because no extra memory per voxel is required. Lastly, this method can enhance the raymarching algorithms by skipping traveral over large pieces of space by recognizing a cluster is empty.
