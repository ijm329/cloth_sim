import 'ebb'
local L = require 'ebblib'
local vdb       = require 'ebb.lib.vdb'

--particle initialization 
local N = 32
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

local function init_pos(idx)
  local i = math.floor(idx / num_particles_width)
  local j = idx % num_particles_width
  local x = i / N
  local z = j / N
  x = BOUND_LENGTH*x + MIN_BOUND
  z = BOUND_LENGTH*z + MIN_BOUND
  return {x, 1, z}
end

particles.pos:Load(init_pos)
particles.prev_pos:Load(particles.pos)

--local particle_positions = {}
--local prev_positions = {}
--for zi=0,num_particles_height - 1 do
--  for xi=0,num_particles_width - 1 do
--    particle_positions[zi*N + xi + 1] = {xi, 1, zi}
--    prev_positions[zi*N + xi + 1] = {xi, 1, zi}
--  end
--end
--
--
--local springs = {}
--
--
----edge field initialization 
--edges:NewField('rest_length', L.float)
--edges:NewField('k', L.float)
--edges:NewField('damping', L.float)
--edges:NewField('type', L.int)
--edges:NewField('left', particles)
--edges:NewField('right', particles)
--
-------------------------------------------------------------------------------

local ebb visualize_particles ( p : particles )
  vdb.color({ 0, 0.5, 0.8 })
  var p2 = p.prev_pos
  vdb.point({ p2[0], p2[1], p2[2] })
end

-------------------------------------------------------------------------------

for i=1,360 do
  vdb.vbegin()
  vdb.frame()
    particles:foreach(visualize_particles)
  vdb.vend()
end
