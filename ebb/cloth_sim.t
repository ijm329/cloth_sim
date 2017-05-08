import 'ebb'
local L = require 'ebblib'
local vdb       = require 'ebb.lib.vdb'

--particle initialization 
local N = 3
local MIN_BOUND = -1.0
local MAX_BOUND = 1.0
local BOUND_LENGTH = MAX_BOUND - MIN_BOUND
local H_LEN = BOUND_LENGTH/N
local D_LEN = H_LEN * L.sqrt(2)
local num_particles_width = N
local num_particles_height = N

--simulation constants
local K = L.Constant(L.float, 100.0)
local GRAVITY = L.Constant(L.vec3f, {0, -9.81, 0})
local WIND = L.Constant(L.vec3f, {0, 0, 1.5})
local DAMPING = L.Constant(L.float, 0.0175)
local STRETCH_CRITICAL = L.Constant(L.float, 1.1)
local TIME_STEP = L.Constant(L.float, 0.00314)
local NUM_CONSTRAINT_ITERS = L.Constant(L.int, 300)
local PARTICLE_MASS = L.Constant(L.float, 0.01)
local H_LEN = L.Constant(L.float, H_LEN)
local D_LEN = L.Constant(L.float, D_LEN)

local particles = L.NewRelation {
  name = "particles",
  size = num_particles_width * num_particles_height,
}

--particle field initialization
particles:NewField('pos', L.vec3f)
particles:NewField('prev_pos', L.vec3f)
particles:NewField('force', L.vec3f):Load({0,0,0})

--particle connectivity initialization
particles:NewField('struct_r', particles)
particles:NewField('struct_d', particles)
particles:NewField('struct_u', particles)
particles:NewField('struct_l', particles)

local ebb init_pos(p : particles)
  var idx = L.id(p)
  var i = idx / N
  var j = idx % N
  var x = L.float(j) / N
  var z = L.float(i) / N
  x = BOUND_LENGTH*x + MIN_BOUND
  z = BOUND_LENGTH*z + MIN_BOUND
  p.pos = L.vec3f({x, 1.0, z})
  p.prev_pos = p.pos
end

local struct_r_keys = {}
local struct_d_keys = {}
local struct_u_keys = {}
local struct_l_keys = {}

for i=0,N-1 do
  for j=0,N-1 do

    if (j ~= N-1) then
      struct_r_keys[i*N+j+1] = (i*N+j+1)
    else
      struct_r_keys[i*N+j+1] = -1
    end

    if(i ~= N-1) then
      struct_d_keys[i*N+j+1] = (i+1)*N + j
    else
      struct_d_keys[i*N+j+1] = -1
    end

    if(i ~= 0) then
      struct_u_keys[i*N+j+1] = (i-1)*N + j
    else
      struct_u_keys[i*N+j+1] = -1
    end

    if(j ~= 0) then
      struct_l_keys[i*N+j+1] = i*N+j-1
    else
      struct_l_keys[i*N+j+1] = -1
    end
  end
end

local ebb debug_connectivity(p:particles)
  var idx = L.id(p)
  var row = idx / N
  var col = idx % N
  L.print(idx)
  --L.print(row, col, p.pos)
  L.print(L.id(p.struct_r))
  L.print(L.id(p.struct_d))
  L.print(L.id(p.struct_l))
  L.print(L.id(p.struct_u))
end

--load positions
particles:foreach(init_pos)

--load spring connections for each particle
particles.struct_r:Load(struct_r_keys)
particles.struct_d:Load(struct_d_keys)
particles.struct_u:Load(struct_u_keys)
particles.struct_l:Load(struct_l_keys)

--particles:foreach(debug_connectivity)

-------------------------------------------------------------------------------

local ebb compute_normal(p1, p2, p3)
  var v1 = p2 - p3
  var v2 = p3 - p1
  return L.vec3f({v1[1]*v2[2] - v1[2]*v2[1], -(v1[0]*v2[2] - v1[2]*v2[0]),
                  v1[0]*v2[1] - v1[1]*v2[0]})
end

local ebb apply_wind_forces(p:particles)
  var idx = L.id(p)
  var row = L.int(idx / N)
  var col = L.int(idx % N)

  --L.print(idx)
  var curr = p.pos
  var force = {0.0, 0.0, 0.0}
  var norm = {0.0, 0.0, 0.0}
  if(col ~= N-1 and row ~= N-1) then
    var right = p.struct_r.pos
    var lower = p.struct_d.pos
    norm = compute_normal(right, curr, lower)
    var wind_force = norm * L.dot(norm / L.length(norm), WIND)
    --L.print(7, idx, norm, wind_force)
    force += wind_force
    --if(row == 0 and col == 1) then L.print(wind_force, norm) end
  end

  if(row ~= N-1 and col ~= 0) then
    var lower = p.struct_d.pos 
    var lower_left = p.struct_d.struct_l.pos
    norm = compute_normal(lower, curr, lower_left)
    var wind_force = norm * L.dot(norm / L.length(norm), WIND)
    force += wind_force
    --L.print(8, idx, norm, wind_force)
    --if(row == 0 and col == 1) then L.print(wind_force, norm) end
  end

  if(row ~= 0 and col ~= N-1) then
    var upper = p.struct_u.pos
    var upper_right = p.struct_u.struct_r.pos
    var right = p.struct_r.pos
    norm = compute_normal(upper_right, upper, curr)
    var wind_force = norm * L.dot(norm / L.length(norm), WIND)
    force += wind_force
    --L.print(9, idx, norm, wind_force)

    norm = compute_normal(right, upper_right, curr)
    wind_force = norm * L.dot(norm / L.length(norm), WIND)
    force += wind_force
    --L.print(10, idx, norm, wind_force)
  end

  if(row ~=0 and col ~= 0) then
    var upper = p.struct_u.pos
    var left = p.struct_l.pos
    norm = compute_normal(curr, upper, left)
    var wind_force = norm * L.dot(norm / L.length(norm), WIND)
    force += wind_force
    --L.print(11, idx, norm, wind_force)
  end

  if(row ~= N-1 and col ~= 0) then
    var left = p.struct_l.pos
    var lower_left = p.struct_d.struct_l.pos
    norm = compute_normal(curr, left, lower_left)
    var wind_force = norm * L.dot(norm / L.length(norm), WIND)
    force += wind_force
    --L.print(12, idx, norm, wind_force)
  end
  p.force += L.vec3f(force)
end

local ebb get_velocity(p)
  var ret = (p.pos - p.prev_pos) / TIME_STEP
  return L.vec3f(ret)
end

local ebb apply_spring_force(p1 : particles, p2 : particles, len : L.float)
  var dir = p2.pos - p1.pos
  var rest = len * (dir / L.float(L.length(dir)))
  var disp = dir - rest
  var vel = get_velocity(p2) - get_velocity(p1)
  var force = -K*disp - DAMPING*vel
  p1.force += L.vec3f(-force)
end

local ebb apply_spring_forces(p:particles)
  var idx = L.id(p)
  var row = L.int(idx / N)
  var col = L.int(idx % N)

  if(col ~= 0) then apply_spring_force(p, p.struct_l, H_LEN) end
  if(row ~= N-1) then apply_spring_force(p, p.struct_d, H_LEN) end
  if(col ~= N-1) then apply_spring_force(p, p.struct_r, H_LEN) end
  if(row ~= 0) then apply_spring_force(p, p.struct_u, H_LEN) end

  if(row ~= 0 and col ~= 0) then apply_spring_force(p, p.struct_u.struct_l, D_LEN) end
  if(row ~= 0 and col ~= N-1) then apply_spring_force(p, p.struct_u.struct_r, D_LEN) end
  if(row ~= N-1 and col ~= 0) then apply_spring_force(p, p.struct_d.struct_l, D_LEN) end
  if(row ~= N-1 and col ~= N-1) then apply_spring_force(p, p.struct_d.struct_r, D_LEN) end

  if((col+2) < N) then apply_spring_force(p, p.struct_r.struct_r, 2*H_LEN) end
  if((col-2) >= 0) then apply_spring_force(p, p.struct_l.struct_l, 2*H_LEN) end
  if((row+2) < N) then apply_spring_force(p, p.struct_d.struct_d, 2*H_LEN) end
  if((row-2) >= 0) then apply_spring_force(p, p.struct_u.struct_u, 2*H_LEN) end
end

local ebb apply_forces(p:particles)
  p.force = PARTICLE_MASS * GRAVITY
  apply_wind_forces(p)

  --spring
  --apply_spring_forces(p)

  L.print(L.id(p), p.force)

end

local ebb update_pos(p:particles)
  var temp = p.pos
  var acc = p.force / PARTICLE_MASS
  p.pos += (p.pos - p.prev_pos + acc * TIME_STEP * TIME_STEP)
  p.prev_pos = temp
end

local ebb reset_fixed_particles(p)
  var idx = L.id(p)
  var row = idx / N
  var col = idx % N
  if((row == 0 and col == 0) or (row == 0 and col == N-1)) then
    var x = L.float(col) / N
    var z = L.float(row) / N
    x = BOUND_LENGTH * x + MIN_BOUND
    z = BOUND_LENGTH * z + MIN_BOUND
    p.pos = L.vec3f({x, 1.0, z})
  end
end

local ebb satisfy_constraints(p:particles)
  reset_fixed_particles(p)
end

-------------------------------------------------------------------------------

local ebb visualize_particles ( p : particles )
  vdb.color({ 0, 0.5, 0.8 })
  var p2 = p.pos
  var idx = L.id(p)
  var row = idx / N
  var col = idx % N
  if((row == 0 and col == 0) or (row == 0 and col == N-1)) then
    vdb.color({0.0f, 1.0f, 0.0f})
  end
  vdb.point({ p2[0], p2[1], p2[2] })

  --vdb.color({1.0, 1.0, 1.0})
  --if(row ~= N-1 and col ~= N-1) then
  --    var p3 = p.struct_d.struct_r
  --    vdb.line({p2[0], p2[1], p2[2]},
  --             {p3.pos[0], p3.pos[1], p3.pos[2]})
  --end

  --if(row ~= 0 and col ~= N-1) then
  --    var p3 = p.struct_u.struct_r.pos
  --    vdb.line({p2[0], p2[1], p2[2]},
  --             {p3[0], p3[1], p3[2]})
  --end

  --vdb.color({1.0, 0.0, 0.0})
  --if(col ~= N-1) then
  --  var p3 = p.struct_r.pos
  --  vdb.line({p2[0], p2[1], p2[2]},
  --           {p3[0], p3[1], p3[2]})
  --end

  --if(row ~= N-1) then
  --  var p3 = p.struct_d.pos
  --  vdb.line({p2[0], p2[1], p2[2]},
  --           {p3[0], p3[1], p3[2]})
  --end

end

-------------------------------------------------------------------------------

for i=1,10 do
  particles:foreach(apply_forces)
  particles:foreach(update_pos)
  particles:foreach(satisfy_constraints)

  vdb.vbegin()
  vdb.frame()
    particles:foreach(visualize_particles)
  vdb.vend()

  print("iter", i)

end
