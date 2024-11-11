# Charles Gil, 260 970 950

import numpy as np

class HalfEdge:
	def __init__(self):
		self.n = None # next
		self.o = None # opposite
		self.v = None # head vertex
		self.f = None # face
		self.child1 = None # first child half edge
		self.child2 = None # second child half edge
		self.parent = None # parent half edge

	def get_curve_nodes(self):
		# get positions for drawing this half edge with polysope
		nodes = np.zeros((3,3))
		nodes[0] = self.n.n.v.p * 0.90 + self.v.p * 0.05 + self.n.v.p * 0.05
		nodes[1] = self.n.n.v.p * 0.05 + self.v.p * 0.90 + self.n.v.p * 0.05
		nodes[2] = self.n.n.v.p * 0.05 + self.v.p * 0.80 + self.n.v.p * 0.15
		return nodes

class Vertex:
	def __init__(self, p):
		self.p = p # position of the point
		self.n = None # normal of the limit surface
		self.he = None # a half edge that points to this vertex
		self.child = None # child vertex

class Face:
	def __init__(self, he):
		self.he = he

class HEDS:
	def	__init__(self, V, F):
		self.verts = []
		self.faces = []
		self.edges = {}
		if len(V)==0: return

		# For each vertex, create a vertex object
		for v in V:
			self.verts.append(Vertex(v))  # Should have same indices as V

		for f in F:
			face = None
			face_hes = []  # List of half edges for the current face
			for vidx in range(len(f)):

				# Create half edge
				he = HalfEdge()
				face_hes.append(he)
				base_vertex_idx = f[(vidx - 1) % len(f)]
				head_vertex_idx = f[vidx]
				self.edges[(base_vertex_idx, head_vertex_idx)] = he

				if vidx == 0:
					face = Face(he)
					self.faces.append(face)  # Create one face object per face

				vertex = self.verts[f[vidx]]
				vertex.he = he  # Link vertex to half edge pointing to it
				he.v = vertex  # Create reference to vertex
				he.f = face  # Create reference to face

			# Connect half edges in a face (next)
			for i in range(len(face_hes)):
				face_hes[i].n = face_hes[(i + 1) % len(f)]

		# Connect opposites
		for v1, v2 in self.edges:
			if (v2, v1) in self.edges:  # Check for opposite half-edge
				he1 = self.edges[(v1, v2)]
				he2 = self.edges[(v2, v1)]
				he1.o = he2
				he2.o = he1

	def get_even_verts(self):
		# get positions for drawing even vertices with polyscope
		if len(self.verts)==0: return []
		if self.verts[0].child is None: return []
		return np.array([v.child.p for v in self.verts])

	def get_odd_verts(self):
		# get positions for drawing odd vertices with polyscope
		if len(self.faces)==0: return []
		if self.faces[0].he.child1 is None: return []
		odd_verts = [ [f.he.child1.v.p for f in self.faces], [f.he.n.child1.v.p for f in self.faces], [f.he.n.n.child1.v.p for f in self.faces] ]
		odd_verts = np.array(odd_verts).reshape(-1, 3)
		return odd_verts

	def get_mesh(self):
		# get positions and faces for drawing the mesh with polysope
		for i,v in enumerate(self.verts): v.ix = i # assign an index to each vertex
		V = np.array([v.p for v in self.verts])
		F = np.array([[f.he.v.ix, f.he.n.v.ix, f.he.n.n.v.ix] for f in self.faces])
		return V, F
	
	def get_limit_normals(self):
		# get the limit normals, if they were computed, otherwise returns nothing
		if len(self.verts)==0: return []
		if self.verts[0].n is None: return []
		return np.array([v.n for v in self.verts])