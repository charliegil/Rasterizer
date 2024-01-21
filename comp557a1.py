# Charles Gil
# 260 970 950

import math
from time import sleep

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
pix = np.zeros((width, height, 3), dtype=np.float32)  # Creates an array of zeros with size of off-screen framebuffer,
# 3 being for colors
depth = np.zeros((width, height, 1), dtype=np.float32)
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))  # off-screen pixels
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width * px, height * px))  # on-screen pixels
V, _, N, T, _, TN = igl.read_obj(args.file)  # read mesh with normals, V=vertex coordinates, N=vertex


# normals,


# T=texture coordinates, TN=texture normals -> does this mean that for each triangle, there is one normal thus 3
# vertices will have the same normal?


@ti.kernel
# copy pixels from small framebuffer to large framebuffer
def copy_pixels():
    for i, j in pixels:
        if px < 2 or (tm.mod(i, px) != 0 and tm.mod(j, px) != 0):
            pixels[i, j] = pixti[i // px, j // px]


def get_bary(triangle):
    p1 = triangle[0]
    p2 = triangle[1]
    p3 = triangle[2]


gui = ti.GUI("Rasterizer", res=(width * px, height * px))

t = 0  # time step for time varying transformations
translate = np.array([width / 2, height / 2, 0])  # translate to center of window
scale = 200 / px * np.eye(3)  # scale to fit in the window

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
    Vt = (scale @ Ry @ Rx @ Rz @ V.T).T  # apply scale and rotation to vertices
    Vt = Vt + translate  # translate vertices
    Nt = (Ry @ Rx @ Rz @ N.T).T  # apply rotation on vertex normals

    # # Compute bounding boxes for all faces
    # for i in range(len(T)):
    #     vertices = list(T[i])  # get indices of vertices associated with face i
    #     triangle = np.array([Vt[vertices[0]], Vt[vertices[1]], Vt[vertices[2]]])
    #     xs = np.array([triangle[0][0], triangle[1][0], triangle[2][0]])
    #     ys = np.array([triangle[0][1], triangle[1][1], triangle[2][1]])
    #     xl = int(min(xs))
    #     xh = int(max(xs))
    #     yl = int(min(ys))
    #     yh = int(max(ys))
    #
    #     # random color
    #     color = list(np.random.choice(range(256), size=3))
    #
    #     # Barycentric coordinates
    #     # igl.barycentric_coordinates_tri(p, triangle[0], triangle[1], triangle[2])
    #
    #     for x in range(xl, xh):
    #         for y in range(yl, yh):
    #
    #             p = (x, y)
    #             # Q1
    #             if args.test == 1:
    #                 pix[x, y] = color
    for x, y in Vt:
        xl = x - 1
        xh = x + 1
        yl = y - 1
        yh = y + 1

    pixti.from_numpy(pix)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001
