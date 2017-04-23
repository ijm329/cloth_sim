#include <iostream>
#include "vector3D.h"

std::ostream& operator<<(std::ostream& os, const vector3D &v)
{
    os << v.x << " " << v.y << " " << v.z << " ";
    return os;
}

vector3D operator*(const float& s, const vector3D& v)
{
    return vector3D(s*v.x, s*v.y, s*v.z);
}
