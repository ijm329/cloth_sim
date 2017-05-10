#ifndef FLOAT3_H 
#define FLOAT3_H 

#include <iostream>
#include <math.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

// negate
inline CUDA_HOSTDEV float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// addition
inline CUDA_HOSTDEV float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CUDA_HOSTDEV void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline CUDA_HOSTDEV float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline CUDA_HOSTDEV void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline CUDA_HOSTDEV float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline CUDA_HOSTDEV float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline CUDA_HOSTDEV float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline CUDA_HOSTDEV void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline CUDA_HOSTDEV float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline CUDA_HOSTDEV float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline CUDA_HOSTDEV float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline CUDA_HOSTDEV void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// dot product
inline CUDA_HOSTDEV float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline CUDA_HOSTDEV float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline CUDA_HOSTDEV float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline CUDA_HOSTDEV float3 normalize(float3 v)
{
    float invLen = 1.0f/sqrtf(dot(v, v));
    return v * invLen;
}

#endif /* FLOAT3_H */
