# Charles Gil, 260 970 950

import numpy as np
import argparse
import igl
import polyscope as ps
import polyscope.imgui as psim
from heds import *
from loop import loop_subdivide

def main():
	global current_he

	psim.TextUnformatted('Charlie Gil - 260970950')
	if psim.Button('next') and current_he is not None:
		current_he = current_he.n
		ps_he.update_node_positions( current_he.get_curve_nodes() )
	psim.SameLine()
	if psim.Button('opposite') and current_he is not None:
		if current_he.o is not None:
			current_he = current_he.o
			ps_he.update_node_positions( current_he.get_curve_nodes() )        
	if psim.Button('child 1') and current_he is not None:
		if current_he.child1 is not None:
			current_he = current_he.child1
			ps_he.update_node_positions( current_he.get_curve_nodes() )
	psim.SameLine()
	if psim.Button('child 2') and current_he is not None:
		if current_he.child2 is not None:
			current_he = current_he.child2
			ps_he.update_node_positions( current_he.get_curve_nodes() )
	if psim.Button('parent') and current_he is not None:
		if current_he.parent is not None:
			current_he = current_he.parent
			ps_he.update_node_positions( current_he.get_curve_nodes() )


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='data/torus.obj')
parser.add_argument('-l', '--levels', type=int, default=2)
args = parser.parse_args()

ps.init()
ps.set_user_callback(main)
ps.set_ground_plane_mode('none')

V, _, _, F, _, _ = igl.read_obj(args.file)
F = np.reshape(F,(-1,3)) # fix faces shape for when we only have 1 face (triangle.obj))
ps_mesh = ps.register_curve_network('L0 Loaded', V, igl.edges(F), enabled=True, color=(1, 0, 1))
heds = [ HEDS(V, F) ]
VS,FS = heds[0].get_mesh()
if len(VS) > 0:
	heds[0].ps_mesh = ps.register_surface_mesh('L0 HEDS', VS, FS, enabled=True, color=(0.5, 0.5, 1),transparency=0.3)

for i in range(1,args.levels+1):
	heds.append( loop_subdivide( heds[i-1] ) )
	even_V = heds[i-1].get_even_verts()
	if len(even_V) > 0: heds[i-1].ps_even = ps.register_point_cloud('L'+str(i)+' Even Verts', even_V, enabled=False, color=(1, 0, 0), radius=0.01)  # draw evens as red
	odd_V = heds[i-1].get_odd_verts()
	if len(odd_V) > 0: heds[i-1].ps_odd = ps.register_point_cloud('L'+str(i)+' Odd Verts', odd_V, enabled=False, color=(0, 1, 0), radius=0.01)  # draw odds as green
	VS,FS = heds[i].get_mesh()
	if len(VS) > 0:
		heds[i].ps_mesh = ps.register_surface_mesh('L'+str(i)+' Subdivided', VS, FS, enabled=False, color=(0.5, 0.5, 1))
		heds[i].ps_mesh.set_back_face_policy('custom')
		heds[i].ps_mesh.set_back_face_color((1,0.5,0.5))
		N = heds[i].get_limit_normals()
		if len(N) > 0: heds[i].ps_mesh.add_vector_quantity('L'+str(i)+' Normals', N, enabled=True)

if hasattr( heds[1], 'ps_mesh' ) : heds[1].ps_mesh.set_enabled(True)
if hasattr( heds[0], 'ps_even' ) : heds[0].ps_even.set_enabled(True)
if hasattr( heds[0], 'ps_odd' ) : heds[0].ps_odd.set_enabled(True)
# if hasattr( heds[-1], 'ps_mesh' ) : heds[-1].ps_mesh.set_smooth_shade(True) # smooth shading for the finest level mesh

if len(heds[0].faces)>0: 
	current_he = heds[0].faces[0].he
	nodes = current_he.get_curve_nodes()
	edges = np.array([[0, 1], [1, 2]])
	ps_he = ps.register_curve_network('Half-Edge', nodes, edges, enabled=True, color=(1, 1, 0), radius=0.002)

ps.show()