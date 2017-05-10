#include "float3.h"

std::ostream& operator<<(std::ostream& os, const float3 &v)
{
    os << "[ " << v.x << " " << v.y << " " << v.z << " ] ";
    return os;
}

vector3D operator*(const float& s, const float3& v)
{
    return vector3D(s*v.x, s*v.y, s*v.z);
}
