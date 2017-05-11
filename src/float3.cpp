#include "float3.h"

std::ostream& operator<<(std::ostream& os, const float3 &v)
{
    os << "[ " << v.x << " " << v.y << " " << v.z << " ] ";
    return os;
}

float3 operator*(const float& s, const float3& v)
{
    return float3(s*v.x, s*v.y, s*v.z);
}
