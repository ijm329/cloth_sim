-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------

import 'ebb'
local L = require 'ebblib'

--particle initialization 

local vdb       = require 'ebb.lib.vdb'
local N = 5
local num_particles_width = N
local num_particles_height = N
local num_shear = 2 * (num_particles_width - 1) * (num_particles_height - 1)
local num_structural = num_particles_height * (num_particles_width - 1) + 
                       num_particles_width * (num_particles_height - 1)
local num_flexion = num_particles_height * (num_particles_width - 2) + 
                    num_particles_width * (num_particles_height - 2)
local num_springs = num_shear + num_structural + num_flexion

local particles = L.NewRelation {
  name = "particles",
  size = num_particles_width * num_particles_height,
}

local edges = L.NewRelation {
  name = "edges",
  -- must account for flexion, shear, and structural springs. 
  size = num_shear + num_structural + num_flexion,
} 



local particle_positions = {}
local prev_positions = {}
for zi=0,num_particles_height - 1 do
  for xi=0,num_particles_width - 1 do
    particle_positions[zi*N + xi + 1] = {xi, 1, zi}
    prev_positions[zi*N + xi + 1] = {xi, 1, zi}
  end
end

--particle field initialization
particles:NewField('pos', L.vec3d):Load(particle_positions)
particles:NewField('prev_pos', L.vec3d):Load(prev_positions)
particles:NewField('force', L.vec3d):Load({0,0,0})
particles:NewField('normal', L.vec3d):Load({0,0,0})
particles:NewField('fixed', L.bool):Load(false)

local springs = {}


--edge field initialization 
edges:NewField('rest_length', L.float)
edges:NewField('k', L.float)
edges:NewField('damping', L.float)
edges:NewField('type', L.int)
edges:NewField('left', particles)
edges:NewField('right', particles)

-------------------------------------------------------------------------------

local ebb visualize_particles ( p : particles )
  vdb.color({ 0, 0.5, 0.8 })
  var p2 = p.pos
  vdb.point({ p2[0], p2[1], p2[2] })
end

-------------------------------------------------------------------------------

for i=1,360 do

  --grid.locate_in_cells(particles, 'pos', 'cell')

  vdb.vbegin()
  vdb.frame()
    particles:foreach(visualize_particles)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
