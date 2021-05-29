#pragma once

#include "Image.h"

#include <string>

class ImageWriter
{
public:
	bool writeImage(const std::string& filename, unsigned char* image, int width, int height, int colorChannels);
	bool writeImage(const std::string& filename, const Image& img);
};