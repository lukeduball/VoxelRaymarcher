#pragma once

#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

template <typename T>
class Vector3
{
public:
	__host__ __device__ Vector3() {}
	__host__ __device__ Vector3(T x, T y, T z) 
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}

	__host__ __device__ inline T getX() const { return data[0]; }
	__host__ __device__ inline T getY() const { return data[1]; }
	__host__ __device__ inline T getZ() const { return data[2]; }

	__host__ __device__ inline const Vector3<T>& operator+() const { return *this; };
	__host__ __device__ inline Vector3<T> operator-() const { return Vector3(-data[0], -data[1], -data[2]); };
	__host__ __device__ inline T operator[](int i) const { return data[i]; }
	__host__ __device__ inline T& operator[](int i) { return data[i]; }
	__host__ __device__ bool operator==(const Vector3<T>& other) const { return other.data[0] == data[0] && other.data[1] == data[1] && other.data[2] == data[2]; };

	__host__ __device__ inline Vector3<T>& operator+=(const Vector3<T>& vector) 
	{ 
		data[0] += vector.data[0];
		data[1] += vector.data[1];
		data[2] += vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3<T>& operator-=(const Vector3<T>& vector)
	{
		data[0] -= vector.data[0];
		data[1] -= vector.data[1];
		data[2] -= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3<T>& operator*=(const Vector3<T>& vector)
	{
		data[0] *= vector.data[0];
		data[1] *= vector.data[1];
		data[2] *= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3<T>& operator/=(const Vector3<T>& vector)
	{
		data[0] /= vector.data[0];
		data[1] /= vector.data[1];
		data[2] /= vector.data[2];
		return *this;
	}

	__host__ __device__ inline Vector3<T>& operator*=(const T t)
	{
		data[0] *= t;
		data[1] *= t;
		data[2] *= t;
		return *this;
	}

	__host__ __device__ inline Vector3<T>& operator/=(const T t)
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

	T data[3];
};

template <typename T>
inline std::istream& operator>>(std::istream& is, Vector3<T>& vector) {
	is >> vector.data[0] >> vector.data[1] >> vector.data[2];
	return is;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& vector) {
	os << vector.data[0] << " " << vector.data[1] << " " << vector.data[2];
	return os;
}

template <typename T>
__host__ __device__ inline Vector3<T> operator+(const Vector3<T>& vector1, const Vector3<T>& vector2) 
{
	return Vector3<T>(vector1.data[0] + vector2.data[0], vector1.data[1] + vector2.data[1], vector1.data[2] + vector2.data[2]);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator-(const Vector3<T>& vector1, const Vector3<T>& vector2)
{
	return Vector3<T>(vector1.data[0] - vector2.data[0], vector1.data[1] - vector2.data[1], vector1.data[2] - vector2.data[2]);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator*(const Vector3<T>& vector1, const Vector3<T>& vector2) 
{
	return Vector3<T>(vector1.data[0] * vector2.data[0], vector1.data[1] * vector2.data[1], vector1.data[2] * vector2.data[2]);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator/(const Vector3<T>& vector1, const Vector3<T>& vector2)
{
	return Vector3<T>(vector1.data[0] / vector2.data[0], vector1.data[1] / vector2.data[1], vector1.data[2] / vector2.data[2]);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator*(float t, const Vector3<T>& v)
{
	return Vector3<T>(t * v.data[0], t * v.data[1], t * v.data[2]);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator/(Vector3<T> v, float t)
{
	return Vector3<T>(v.data[0] / t, v.data[1] / t, v.data[2] / t);
}

template <typename T>
__host__ __device__ inline Vector3<T> operator*(const Vector3<T>& v, float t)
{
	return Vector3<T>(t * v.data[0], t * v.data[1], t * v.data[2]);
}

template <typename T>
__host__ __device__ inline float dot(const Vector3<T>& vector1, const Vector3<T>& vector2)
{
	return vector1.data[0] * vector2.data[0] + vector1.data[1] * vector2.data[1] + vector1.data[2] * vector2.data[2];
}

template <typename T>
__host__ __device__ inline Vector3<T> cross(const Vector3<T>& vector1, const Vector3<T>& vector2)
{
	return Vector3<T>((vector1.data[1] * vector2.data[2] - vector1.data[2] * vector2.data[1]),
		(-(vector1.data[0] * vector2.data[2] - vector1.data[2] * vector2.data[0])),
		(vector1.data[0] * vector2.data[1] - vector1.data[1] * vector2.data[0]));
}

template <typename T>
__host__ __device__ inline Vector3<T> makeUnitVector(const Vector3<T>& vector)
{
	return vector / vector.length();
}

typedef Vector3<float> Vector3f;
typedef Vector3<int32_t> Vector3i;

class Vector3iHashFunction
{
public:
	//Cantor Pairing Hash function
	size_t operator()(const Vector3i& vec3i) const
	{
		size_t hash1 = 0.5 * (vec3i.data[0] + vec3i.data[1]) * (vec3i.data[0] + vec3i.data[1] + 1) + vec3i.data[1];
		size_t hash2 = 0.5 * (hash1 + vec3i.data[2]) * (hash1 + vec3i.data[2] + 1) + vec3i.data[2];
		return hash2;
	}
};