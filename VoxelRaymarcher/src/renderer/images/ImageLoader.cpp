#include "ImageLoader.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>

Image ImageLoader::loadImage(const std::string& path)
{
	Image img;
	img.image = stbi_load(path.c_str(), &img.width, &img.height, 0, STBI_rgb_alpha);
	if (img.image == nullptr)
	{
		std::cout << "WARNING: Could not find image at path: " << path << std::endl;
	}
	img.colorChannels = 4;
	return img;
}
