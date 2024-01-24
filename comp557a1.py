# Charles Gil
# 260 970 950

import math
import igl
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="cube.obj")  # Add argument requires an input for
# each added argument when called from the command line
parser.add_argument("--width", type=int, default=1440, help="Width of off screen framebuffer")
parser.add_argument("--height", type=int, default=720, help="Height of off screen framebuffer")
parser.add_argument("--px", type=int, default=10, help="Size of pixel in on screen framebuffer")
parser.add_argument("--test", type=int, help="run a numbered unit test")
args = parser.parse_args()
ti.init(arch=ti.cpu)  # can also use ti.gpu -> tells program to run on cpu
px = args.px  # Size of pixel in on screen framebuffer
width, height = args.width // px, args.height // px  # Size of off-screen framebuffer
pix = np.zeros((width, height, 3), dtype=np.float32)  # Pixel data with rgb values per pixel
depth = np.zeros((width, height, 1), dtype=np.float32)  # Pixel data with z depth value per pixel
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))  # off-screen pixels
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width * px, height * px))  # on-screen pixels
V, _, N, T, _, TN = igl.read_obj(args.file)  # read mesh with normals, V=vertex coordinates, N=vertex


@ti.kernel
# copy pixels from small framebuffer to large framebuffer
def copy_pixels():
    for i, j in pixels:
        if px < 2 or (tm.mod(i, px) != 0 and tm.mod(j, px) != 0):
            pixels[i, j] = pixti[i // px, j // px]


# Generates random color, 3 values of float from 0 to 1
def random_color():
    r = np.random.uniform(0, 1)
    g = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    return [r, g, b]


def compute_normals():
    for fidx in T:
        points = [Vt[fidx][i] for i in range(3)]
        e1 = np.subtract(points[1], points[0])
        e2 = np.subtract(points[2], points[0])
        n = np.cross(e1, e2)  # Warning is a bug in numpy
        n = n / np.linalg.norm(n)
        # Set normal of face and vertices to n
        N[fidx] = n
        T[fidx] = [fidx, fidx, fidx]


# Helper function for computing barycentric coordinates
def fxy(x, y, points, t):
    x0, y0 = (points[0][i] for i in range(2))
    x1, y1 = (points[1][i] for i in range(2))
    x2, y2 = (points[2][i] for i in range(2))

    f = 0

    if t == 1:
        f = (y0 - y1) * x + (x1 - x0) * y + x0 * y1 - x1 * y0
    if t == 12:
        f = (y1 - y2) * x + (x2 - x1) * y + x1 * y2 - x2 * y1
    if t == 20:
        f = (y2 - y0) * x + (x0 - x2) * y + x2 * y0 - x0 * y2

    return f


gui = ti.GUI("Rasterizer", res=(width * px, height * px))
t = 0  # time step for time varying transformations
translate = np.array([width / 2, height / 2, 0])  # translate to center of window
scale = 200 / px * np.eye(3)  # scale to fit in the window

# Generate random color for each face
color = np.zeros((len(T)))
for fidx in T:
    color[fidx] = random_color()

while gui.running:
    pix.fill(0)  # clear pixel buffer
    depth.fill(-math.inf)  # clear depth buffer
    # time varying transformation
    c, s = math.cos(1.2 * t), math.sin(1.2 * t)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = math.cos(t), math.sin(t)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c, s = math.cos(1.8 * t), math.sin(1.8 * t)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    Vt = (scale @ Ry @ Rx @ Rz @ V.T).T
    Vt = Vt + translate

    # Compute normals if undefined
    if N.shape[0] == 0:
        N = np.zeros(len(T))
        compute_normals()

    Nt = (Ry @ Rx @ Rz @ N.T).T

    # Q1 Compute bounding boxes
    for fidx in T:  # for each face

        # Get coordinates of vertices of face
        vertices = [Vt[fidx][i] for i in range(3)]  # get vertex coords for face
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        # Compute coordinates of bounding box
        xl = np.min(xs)
        xl = int(np.floor(xl))
        xh = np.max(xs)
        xh = int(np.ceil(xh))
        yl = np.min(ys)
        yl = int(np.floor(yl))
        yh = np.max(ys)
        yh = int(np.ceil(yh))

        # Clamp to display window
        if xl < 0:
            xl = 0

        if xh > width - 1:
            xh = width - 1

        if yl < 0:
            yl = 0

        if yh > height - 1:
            yh = height - 1

        # Intermediate computation for barycentric coordinates of triangle
        p0, p1, p2 = (vertices[i] for i in range(3))
        fa = fxy(p0[0], p0[1], vertices, 12)
        fb = fxy(p1[0], p1[1], vertices, 20)
        fg = fxy(p2[0], p2[1], vertices, 1)

        # Pixel walk algorithm
        for y in range(yl, yh):
            for x in range(xl, xh):

                # Q1 Color bounding boxes
                if args.test == 1:
                    pix[x][y] = color[fidx]
                    continue

                # Intermediate computations for barycentric coordinates
                alpha = fxy(x, y, vertices, 12) / fa
                beta = fxy(x, y, vertices, 20) / fb
                gamma = fxy(x, y, vertices, 1) / fg

                # Interpolate z value of point on triangle
                uz = p0[2]
                vz = p1[2]
                wz = p2[2]
                z = alpha * uz + beta * vz + gamma * wz

                #############
                indices = TN[fidx]
                normals = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                bary = [alpha, beta, gamma]
                for i in range(3):
                    x_idx, y_idx, z_idx = (indices[i][j] for j in range(3))
                    normals[i] = [N[x_idx][0] * bary[i], N[y_idx][1] * bary[i], N[z_idx][2] * bary[i]]
                normal = np.add(normals[0], normals[1])
                normal = np.add(normal, normals[2])
                normal_z = normal[2]
                #############

                if 0 <= alpha and 0 <= beta and 0 <= gamma:
                    if (alpha > 0 or fa * fxy(-1, -1, vertices, 12) > 0) and (beta > 0 or fb * fxy(-1, -1, vertices, 20) > 0) and (gamma > 0 or fg * fxy(-1, -1, vertices, 1) > 0):

                        # Q2 Draw triangles (no depth)
                        if args.test == 2:
                            pix[x][y] = [alpha, beta, gamma]
                            continue

                        if z > depth[x][y]:
                            depth[x][y] = z

                            if args.test == 3:
                                pix[x][y] = [alpha, beta, gamma]
                                continue

                            if normal_z > 0:
                                r = g = b = normal_z
                            else:
                                r = g = b = 0

                            pix[x][y] = [r, g, b]

    pixti.from_numpy(pix)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001
