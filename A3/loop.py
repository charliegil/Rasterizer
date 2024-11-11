# Charles Gil, 260 970 950

import numpy as np
from heds import *


def loop_subdivide(heds):
    heds.child = HEDS([], [])  # create a new empty half edge data structure
    for v in heds.verts:
        # TODO subdivide the vertex, and append to heds.child.verts (create even vertices)
        even = subdivide_vertex(v.he)
        v.child = even
        heds.child.verts.append(v.child)

    # Dict to keep track of odd edges created
    visited = []
    for f in heds.faces:

        # Get original vertices
        edges = [f.he, f.he.n, f.he.n.n]
        original_vertices = []
        for e in edges:
            original_vertices.append(e.v)

        odds = []

        # Create odd vertices
        for e in edges:
            if e in visited or e.o in visited:  # Make sure not to create duplicate vertices
                continue

            visited.append(e)  # Add e and twin to visited edges to no duplicate vertex
            visited.append(e.o)
            v0 = e.n.n.v
            v1 = e.v
            odd = subdivide_edge(e)
            odds.append(odd)
            heds.child.verts.append(odd)  # Add to list of vertices

            # Create new child half edges of e
            e1 = HalfEdge()  # Edge from v0 to odd
            e1.v = odd
            odd.he = e1
            e1.parent = e

            e2 = HalfEdge()  # Edge from odd to v1
            e2.v = v1.child
            v1.child.he = e2
            e2.parent = e

            # Set child pointers
            e.child1 = e1
            e.child2 = e2

            # Create new child half edges of twin edge
            twin = e.o
            e3 = HalfEdge()
            e3.v = odd
            e3.parent = twin

            e4 = HalfEdge()
            e4.v = v0.child
            v0.child.he = e4
            e4.parent = twin

            twin.child1 = e3
            twin.child2 = e4

            # Connect child twins
            e1.o = e4
            e4.o = e1
            e2.o = e3
            e3.o = e2

        # Get parent half edges of even vertices adjacent to vertex (will create 3 outer faces)
        twins = []
        for vertex in original_vertices:
            e1 = None
            for e in edges:
                if e.v == vertex:
                    e1 = e

            e0 = e1.n
            e1 = e1.o

            child_vertex = vertex.child
            child_e0 = e0.child1
            child_e1 = e1.child1

            # Get reference to even vertices adjacent to child_vertex
            v0 = child_e0.v
            v1 = child_e1.v

            # Create inner edges
            inner = HalfEdge()
            inner.v = v1

            inner_twin = HalfEdge()
            inner_twin.v = v0

            twins.append(inner_twin)

            # Link opposites
            inner.o = inner_twin
            inner_twin.o = inner

            # Link next (can't do for outer edges yet)
            # Inner edges
            inner.n = child_e1.o
            child_e1.o.n = child_e0
            child_e0.n = inner

            face = Face(child_e1)
            heds.child.faces.append(face)

        # Create innermost face
        inner_face = Face(twins[0])
        heds.child.faces.append(inner_face)

        # Link inner half edges together
        for i in range(len(twins)):
            twins[i].n = twins[(i + 1) % len(twins)]

    for v in heds.child.verts:
        compute_limit_normal(v)

    return heds.child


# Compute and create even vertex
def subdivide_vertex(he):
    current = he.n.o
    neighbour_sum = np.zeros(3)
    neighbour_sum += he.o.v.p
    n = 1
    while current != he:
        neighbour_sum += current.o.v.p
        current = current.n.o
        n += 1

    # Warren Rules (had bug for Loop rules with n = 3)
    if n == 3:
        beta = 3 / 16

    else:
        beta = 3 / (8 * n)

    new_pos = (1 - n * beta) * he.v.p + beta * neighbour_sum
    return Vertex(new_pos)


def subdivide_edge(he):
    new_pos = 1 / 8 * (he.n.v.p + he.o.n.v.p) + 3 / 8 * (he.v.p + he.n.n.v.p)
    return Vertex(new_pos)


def compute_limit_normal(v):
    # Get position of adjacent vertices
    points = [v.he.o.v.p]
    current = v.he.n.o

    while current != v.he:
        points.append(current.o.v.p)
        current = current.n.o

    t1 = 0
    t2 = 0

    current = v.he
    k = len(points)
    for i in range(k):
        t2 += np.cos(2 * np.pi * i / k) * points[i]
        t1 += np.sin(2 * np.pi * i / k) * points[i]

    v.n = np.cross(t1, t2) / np.linalg.norm(np.cross(t1, t2))


