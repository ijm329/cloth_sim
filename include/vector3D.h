#ifndef VECTOR3D_H 
#define VECTOR3D_H 

#include <iostream>
#include <math.h>

typedef struct vector3D
{
    float x;
    float y;
    float z;

    vector3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {};
    vector3D(const vector3D& v): x(v.x), y(v.y), z(v.z) {};

    vector3D operator+(const vector3D& a) const
    {
        return vector3D(x+a.x, y+a.y, z+a.z);
    }

    vector3D operator-(const vector3D& a) const
    {
        return vector3D(x-a.x, y-a.y, z-a.z);
    }
    
    vector3D operator*(const float& s) const
    {
        return vector3D(x*s, y*s, z*s);
    }

    vector3D operator/(const float& s) const
    {
        float s_inv = 1.0/s;
        return vector3D(x*s_inv, y*s_inv, z*s_inv);
    }

    vector3D operator-() const
    {
        return vector3D(-x, -y, -z);
    }

    void operator+=(const vector3D& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator-=(const vector3D& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    void operator*=(const float& s)
    {
        x *= s;
        y *= s;
        z *= s;
    }

    vector3D& operator=(const vector3D& a)
    {
        x = a.x;
        y = a.y;
        z = a.z;
        return *this;
    }

    bool operator==(const vector3D& a) const
    {
        return ((x == a.x) && (y == a.y) && (z == a.z));
    }

    bool operator!=(const vector3D& a) const
    {
        return ((x != a.x) || (y != a.y) || (z != a.z));
    }

    float norm() const
    {
        return sqrt(x*x + y*y + z*z);
    }

    vector3D unit() const
    {
        float norm_inv = 1.0f / sqrt(x*x + y*y + z*z);
        return vector3D(norm_inv * x, norm_inv * y, norm_inv * z);
    }

    void normalize()
    {
        float norm_inv = 1.0f / sqrt(x*x + y*y + z*z);
        x *= norm_inv;
        y *= norm_inv;
        z *= norm_inv;
    }
    vector3D cross_product(const vector3D& l2) const
    {
        return vector3D(y * l2.z - z * l2.y, -(x * l2.z - z * l2.x),
        (x * l2.y - y * l2.x));
    }

    float dot_product(const vector3D& l2) const
    {
        float dp;
        dp = x * l2.x + y * l2.y + z * l2.z;
        return dp;
    }

} vector3D;

std::ostream& operator<<(std::ostream& os, const vector3D &v);
vector3D operator*(const float& s, const vector3D& v);

#endif /* VECTOR3D_H */
