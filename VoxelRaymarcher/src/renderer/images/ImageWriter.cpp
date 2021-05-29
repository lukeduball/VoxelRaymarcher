#include "ImageWriter.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <iostream>

bool ImageWriter::writeImage(const std::string& filename, unsigned char* image, int width, int height, int colorChannels)
{
	int result = stbi_write_png(filename.c_str(), width, height, colorChannels, image, 0);
	if (!result)
	{
		std::cout << "ERROR: Failed to write image to: " << filename << std::endl;
	}
	return result;
}

bool ImageWriter::writeImage(const std::string& filename, const Image& img)
{
	return writeImage(filename, img.image, img.width, img.height, img.colorChannels);
}
