#pragma once

#include <ctime>
#include <random>

class Random
{
public:
	static void initialize()
	{
		srand(time(NULL));
	}

	static uint32_t getRandomInt(uint32_t low, uint32_t high)
	{
		uint32_t diff = high - low;
		return rand() % diff + low;
	}
};