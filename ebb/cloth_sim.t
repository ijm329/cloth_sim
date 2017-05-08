import 'ebb'
local L = require 'ebblib'
local vdb       = require 'ebb.lib.vdb'

--particle initialization 
local N = 16
local MIN_BOUND = -1
local MAX_BOUND = 1
local BOUND_LENGTH = MAX_BOUND - MIN_BOUND
local num_particles_width = N
local num_particles_height = N

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
particles:NewField('shear_dr', particles)
particles:NewField('shear_dl', particles)
particles:NewField('flexion_r', particles)
particles:NewField('flexion_d', particles)

local function init_pos(idx)
  local i = math.floor(idx / num_particles_width)
  local j = idx % num_particles_width
  local x = i / N
  local z = j / N
  x = BOUND_LENGTH*x + MIN_BOUND
  z = BOUND_LENGTH*z + MIN_BOUND
  return {x, 1, z}
end

local struct_r_keys = {}
local struct_d_keys = {}
local shear_dr_keys = {}
local shear_dl_keys = {}
local flexion_r_keys = {}
local flexion_d_keys = {}

for i=0,N-1 do
  for j=0,N-1 do

    --structural links
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
    
    --shear links
    if (j == 0) then
      local rr = i+1
      local rc = j+1
      shear_dr_keys[i*N+j+1] = (rr*N + rc)
      shear_dl_keys[i*N+j+1] = -1
    elseif (j == N-1) then
      local rr = i+1
      local rc = j-1
      shear_dl_keys[i*N+j+1] = (rr*N + rc)
      shear_dr_keys[i*N+j+1] = -1
    else
      local rr = i+1
      local rc = j+1
      shear_dr_keys[i*N+j+1] = (rr*N + rc)
      rc = j-1
      shear_dl_keys[i*N+j+1] = (rr*N + rc)
    end

    --flexion links
    if((i+2) < N) then
      flexion_d_keys[i*N+j+1] = (i+2)*N + j
    else
      flexion_d_keys[i*N+j+1] = -1
    end

    if((j+2) < N) then
      flexion_r_keys[i*N+j+1] = i*N+j+2
    else
      flexion_r_keys[i*N+j+1] = -1
    end

  end
end

local ebb apply_forces(idx)
end

--load spring keys
particles.shear_dr:Load(shear_dr_keys)
particles.shear_dl:Load(shear_dl_keys)
particles.struct_r:Load(struct_r_keys)
particles.struct_d:Load(struct_d_keys)
particles.flexion_r:Load(flexion_r_keys)
particles.flexion_d:Load(flexion_d_keys)

--load positions
particles.pos:Load(init_pos)
particles.prev_pos:Load(particles.pos)

-------------------------------------------------------------------------------

local ebb visualize_particles ( p : particles )
  vdb.color({ 0, 0.5, 0.8 })
  var p2 = p.pos
  vdb.point({ p2[0], p2[1], p2[2] })
  var idx = L.id(p)
  var row = idx / N
  var col = idx % N

  --shear springs
  vdb.color({1.0, 1.0, 1.0})
  if(col ~= N-1 and row ~= N-1) then
    vdb.line({ p2[0], p2[1], p2[2] },
             { p.shear_dr.pos[0], p.shear_dr.pos[1], p.shear_dr.pos[2]})
  end
  vdb.color({1.0, 1.0, 1.0})
  if(col ~= 0 and row ~= N-1) then
    vdb.line({ p2[0], p2[1], p2[2] },
             { p.shear_dl.pos[0], p.shear_dl.pos[1], p.shear_dl.pos[2]})
  end

  --structural springs
  if(col ~= N-1) then
    vdb.line({ p2[0], p2[1], p2[2] },
             { p.struct_r.pos[0], p.struct_r.pos[1], p.struct_r.pos[2]})
  end

  if(row ~= N-1) then
    vdb.line({ p2[0], p2[1], p2[2] },
             { p.struct_d.pos[0], p.struct_d.pos[1], p.struct_d.pos[2]})
  end

  vdb.color({1.0, 0.0, 0.0})

  --flexion springs
  --if((row+2) < N) then
  --  vdb.line({ p2[0], p2[1], p2[2] },
  --           { p.flexion_d.pos[0], p.flexion_d.pos[1], p.flexion_d.pos[2]})
  --end

  --if((col+2) < N) then
  --  vdb.line({ p2[0], p2[1], p2[2] },
  --           { p.flexion_r.pos[0], p.flexion_r.pos[1], p.flexion_r.pos[2]})
  --end
end

-------------------------------------------------------------------------------

for i=1,360 do
  vdb.vbegin()
  vdb.frame()
    particles:foreach(apply_forces)
    particles:foreach(visualize_particles)
  vdb.vend()

  if i % 10 == 0 then print ('iter', i) end
end
