#pragma once

#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

class Vector3
{
public:
	__host__ __device__ Vector3() : Vector3(0.0f, 0.0f, 0.0f) {}
	__host__ __device__ Vector3(float x, float y, float z) 
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}

	__host__ __device__ inline float getX() const { return data[0]; }
	__host__ __device__ inline float getY() const { return data[1]; }
	__host__ __device__ inline float getZ() const { return data[2]; }

	__host__ __device__ inline const Vector3& operator+() const { return *this; };
	__host__ __device__ inline Vector3 operator-() const { return Vector3(-data[0], -data[1], -data[2]); };
	__host__ __device__ inline float operator[](int i) const { return data[i]; }
	__host__ __device__ inline float& operator[](int i) { return data[i]; }

	__host__ __device__ inline Vector3& operator+=(const Vector3& vector) 
	{ 
		data[0] += vector.data[0];
		data[1] += vector.data[1];
		data[2] += vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3& operator-=(const Vector3& vector)
	{
		data[0] -= vector.data[0];
		data[1] -= vector.data[1];
		data[2] -= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3& operator*=(const Vector3& vector)
	{
		data[0] *= vector.data[0];
		data[1] *= vector.data[1];
		data[2] *= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3& operator/=(const Vector3& vector)
	{
		data[0] /= vector.data[0];
		data[1] /= vector.data[1];
		data[2] /= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3& operator*=(const float t)
	{
		data[0] *= t;
		data[1] *= t;
		data[2] *= t;
		return *this;
	}

	__host__ __device__ inline Vector3& operator/=(const float t)
	{
		data[0] /= t;
		data[1] /= t;
		data[2] /= t;
		return *this;
	}

	__host__ __device__ inline float length() const { return sqrtf(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]); }
	__host__ __device__ inline float squared_length() const { return data[0] * data[0] + data[1] * data[1] + data[2] * data[2]; }

	__host__ __device__ inline void normalize()
	{
		float fraction = 1.0f / sqrtf(length());
		data[0] * fraction;
		data[1] * fraction;
		data[2] * fraction;
	}

	float data[3];
};

inline std::istream& operator>>(std::istream& is, Vector3& vector) {
	is >> vector.data[0] >> vector.data[1] >> vector.data[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const Vector3& vector) {
	os << vector.data[0] << " " << vector.data[1] << " " << vector.data[2];
	return os;
}

__host__ __device__ inline Vector3 operator+(const Vector3& vector1, const Vector3& vector2) 
{
	return Vector3(vector1.data[0] + vector2.data[0], vector1.data[1] + vector2.data[1], vector1.data[2] + vector2.data[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3& vector1, const Vector3& vector2)
{
	return Vector3(vector1.data[0] - vector2.data[0], vector1.data[1] - vector2.data[1], vector1.data[2] - vector2.data[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3& vector1, const Vector3& vector2) 
{
	return Vector3(vector1.data[0] * vector2.data[0], vector1.data[1] * vector2.data[1], vector1.data[2] * vector2.data[2]);
}

__host__ __device__ inline Vector3 operator/(const Vector3& vector1, const Vector3& vector2)
{
	return Vector3(vector1.data[0] / vector2.data[0], vector1.data[1] / vector2.data[1], vector1.data[2] / vector2.data[2]);
}

__host__ __device__ inline Vector3 operator*(float t, const Vector3& v) 
{
	return Vector3(t * v.data[0], t * v.data[1], t * v.data[2]);
}

__host__ __device__ inline Vector3 operator/(Vector3 v, float t) 
{
	return Vector3(v.data[0] / t, v.data[1] / t, v.data[2] / t);
}

__host__ __device__ inline Vector3 operator*(const Vector3& v, float t) 
{
	return Vector3(t * v.data[0], t * v.data[1], t * v.data[2]);
}

__host__ __device__ inline float dot(const Vector3& vector1, const Vector3& vector2) 
{
	return vector1.data[0] * vector2.data[0] + vector1.data[1] * vector2.data[1] + vector1.data[2] * vector2.data[2];
}

__host__ __device__ inline Vector3 cross(const Vector3& vector1, const Vector3& vector2) 
{
	return Vector3((vector1.data[1] * vector2.data[2] - vector1.data[2] * vector2.data[1]),
		(-(vector1.data[0] * vector2.data[2] - vector1.data[2] * vector2.data[0])),
		(vector1.data[0] * vector2.data[1] - vector1.data[1] * vector2.data[0]));
}

__host__ __device__ inline Vector3 makeUnitVector(const Vector3& vector)
{
	return vector / vector.length();
}