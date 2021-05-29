#pragma once

#include <string>

#include "Image.h"

class ImageLoader
{
public:
	Image loadImage(const std::string& path);
};