#pragma once

#include <cuda_runtime.h>
#include <iostream>

template<typename T>
class Vector2
{
public:
	__host__ __device__ Vector2() {}
	__host__ __device__ Vector2(T x, T y)
	{
		data[0] = x;
		data[1] = y;
	}

	__host__ __device__ inline T getX() const { return data[0]; };
	__host__ __device__ inline T getY() const { return data[1]; };

	__host__ __device__ inline const Vector2<T>& operator+() const { return *this; };
	__host__ __device__ inline Vector2<T> operator-() const { return Vector2(-data[0], -data[1]); };
	__host__ __device__ inline T operator[](int i) const { return data[i]; }
	__host__ __device__ inline T& operator[](int i) { return data[i]; }
	__host__ __device__ bool operator==(const Vector2<T>& other) const { return other.data[0] == data[0] && other.data[1] == data[1]; };

	__host__ __device__ inline Vector2<T>& operator+=(const Vector2<T>& vector)
	{
		data[0] += vector.data[0];
		data[1] += vector.data[1];
		return *this;
	}

	__host__ __device__ inline Vector2<T>& operator-=(const Vector2<T>& vector)
	{
		data[0] -= vector.data[0];
		data[1] -= vector.data[1];
		return *this;
	}

	__host__ __device__ inline Vector2<T>& operator*=(const Vector2<T>& vector)
	{
		data[0] *= vector.data[0];
		data[1] *= vector.data[1];
		return *this;
	}

	__host__ __device__ inline Vector2<T>& operator/=(const Vector2<T>& vector)
	{
		data[0] /= vector.data[0];
		data[1] /= vector.data[1];
		return *this;
	}

	__host__ __device__ inline Vector2<T>& operator*=(const T t)
	{
		data[0] *= t;
		data[1] *= t;
		return *this;
	}

	__host__ __device__ inline Vector2<T>& operator/=(const T t)
	{
		data[0] /= t;
		data[1] /= t;
		return *this;
	}

	__host__ __device__ inline float length() const { return sqrtf(data[0] * data[0] + data[1] * data[1]); }
	__host__ __device__ inline float squared_length() const { return data[0] * data[0] + data[1] * data[1]; }

	__host__ __device__ inline void normalize()
	{
		float fraction = 1.0f / sqrtf(length());
		data[0] * fraction;
		data[1] * fraction;
	}

	T data[2];
};

namespace vec2
{
	template <typename T>
	__host__ __device__ inline float dot(const Vector2<T>& vector1, const Vector2<T>& vector2)
	{
		return vector1.data[0] * vector2.data[0] + vector1.data[1] * vector2.data[1];
	}

	template <typename T>
	__host__ __device__ inline Vector2<T> makeUnitVector(const Vector2<T>& vector)
	{
		return vector / vector.length();
	}
}

template <typename T>
inline std::istream& operator>>(std::istream& is, Vector2<T>& vector) {
	is >> vector.data[0] >> vector.data[1];
	return is;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vector2<T>& vector) {
	os << vector.data[0] << " " << vector.data[1];
	return os;
}

template <typename T>
__host__ __device__ inline Vector2<T> operator+(const Vector2<T>& vector1, const Vector2<T>& vector2)
{
	return Vector2<T>(vector1.data[0] + vector2.data[0], vector1.data[1] + vector2.data[1]);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator-(const Vector2<T>& vector1, const Vector2<T>& vector2)
{
	return Vector2<T>(vector1.data[0] - vector2.data[0], vector1.data[1] - vector2.data[1]);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator*(const Vector2<T>& vector1, const Vector2<T>& vector2)
{
	return Vector2<T>(vector1.data[0] * vector2.data[0], vector1.data[1] * vector2.data[1]);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator/(const Vector2<T>& vector1, const Vector2<T>& vector2)
{
	return Vector2<T>(vector1.data[0] / vector2.data[0], vector1.data[1] / vector2.data[1]);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator*(float t, const Vector2<T>& v)
{
	return Vector2<T>(t * v.data[0], t * v.data[1]);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator/(Vector2<T> v, float t)
{
	return Vector2<T>(v.data[0] / t, v.data[1] / t);
}

template <typename T>
__host__ __device__ inline Vector2<T> operator*(const Vector2<T>& v, float t)
{
	return Vector2<T>(t * v.data[0], t * v.data[1]);
}

typedef Vector2<float> Vector2f;