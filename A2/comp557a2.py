# Charles Gil, 260970950

import moderngl_window as mglw
import moderngl as mgl
import numpy as np
from pyrr import matrix44
import random

from scipy.spatial.transform import Rotation

rotation_type = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ', 'RL', 'QL', 'QS',
                 'QLN', 'QLNF', 'QSF', 'A', 'B']

# dict mapping letters x y z to colors
letter_colors = {
    'X': np.array((1, 0, 0), dtype='f4'),
    'Y': np.array((0, 1, 0), dtype='f4'),
    'Z': np.array((0, 0, 1), dtype='f4'),
    'R': np.array((.3, .7, 0), dtype='f4'),
    'Q': np.array((0, .7, .7), dtype='f4'),
    'L': np.array((.7, 0, .7), dtype='f4'),
    'S': np.array((.7, .3, 0), dtype='f4'),
    'N': np.array((.7, .3, .3), dtype='f4'),
    'F': np.array((.3, .7, .3), dtype='f4'),
    'A': np.array((.7, .7, .7), dtype='f4'),
    'B': np.array((.7, .7, .7), dtype='f4')
}


def normalize(q):
    return q / np.linalg.norm(q)


def rand_unit_quaternion():
    q = np.array([random.gauss(0, 1) for i in range(4)])
    return normalize(q)


def quaternion_random_axis_angle(angle):
    axis = np.array([random.gauss(0, 1) for i in range(3)])
    axis = axis / np.linalg.norm(axis)
    return np.append(np.cos(angle / 2), np.sin(angle / 2) * axis)


def rand_180_quaternion():
    return quaternion_random_axis_angle(np.pi)


def quaternion_multiply(q1, q2):
    q1q2 = np.zeros(4)
    q1q2[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q1q2[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return q1q2


# Converts unit quaternion to rotation matrix
def quat_to_R(q):
    return np.array(
        [[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]])


class Body:
    def __init__(self, pos, rotorder, prog, vao):
        self.target_orientation_A = None  # Added target orientations for interpolating
        self.target_orientation_B = None
        self.rotorder = rotorder
        self.pos = pos
        self.prog = prog
        self.vao = vao

    def set_rotation(self, q, i):
        # Set the target rotation from the provided quaternion, where i is 0 or 1 for target A or B respectively
        if i == 0:
            self.target_orientation_A = q
        elif i == 1:
            self.target_orientation_B = q

    def interpolate_euler(self, t):

        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        r1 = quat_to_R(self.target_orientation_A)
        r2 = quat_to_R(self.target_orientation_B)

        r1 = Rotation.from_matrix(r1)
        r2 = Rotation.from_matrix(r2)

        euler1 = Rotation.as_euler(r1,self.rotorder)
        euler2 = Rotation.as_euler(r2,self.rotorder)

        # Perform linear interpolation for each angle component
        angles = (1 - t) * euler1 + t * euler2
        angles = angles[0:3]

        interpolated_rotation = Rotation.from_euler(self.rotorder, angles).as_matrix()

        return interpolated_rotation

    def interpolate_rl(self, t):

        r1 = quat_to_R(self.target_orientation_A)
        r2 = quat_to_R(self.target_orientation_B)
        interpolated_rotation = [[0, 0,0],
                                 [0,0,0],
                                 [0,0,0]]
        for i in range(len(r1)):
            for j in range(len(r2)):
                interpolated_entry = (1 - t) * r1[i][j] + t * r2[i][j]
                interpolated_rotation[i][j] = interpolated_entry

        return interpolated_rotation

    def interpolate_ql(self, t):
        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        # Linearly interpolate quaternions q1, q2
        interpolated_quaternion = (1 - t) * q1 + t * q2

        interpolated_rotation = quat_to_R(interpolated_quaternion)
        return interpolated_rotation

    def interpolate_qln(self, t):
        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        # Linearly interpolate quaternions q1, q2
        interpolated_quaternion = (1 - t) * q1 + t * q2
        interpolated_quaternion = interpolated_quaternion / np.linalg.norm(interpolated_quaternion)

        interpolated_rotation = quat_to_R(interpolated_quaternion)
        return interpolated_rotation

    def interpolate_qlnf(self, t):
        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        dot_product = np.dot(q1, q2)
        if dot_product < 0:
            q1 = -q1

        # Linearly interpolate quaternions q1, q2
        interpolated_quaternion = (1 - t) * q1 + t * q2
        interpolated_quaternion = interpolated_quaternion / np.linalg.norm(interpolated_quaternion)

        interpolated_rotation = quat_to_R(interpolated_quaternion)
        return interpolated_rotation

    def interpolate_qs(self, t):
        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        theta = np.arccos(np.dot(q1, q2))

        if theta == 0:
            interpolated_quaternion = (1 - t) * q1 + t * q2
        else:
            interpolated_quaternion = np.sin((1 - t) * theta) / np.sin(theta) * q1 + np.sin(t * theta) / np.sin(
                theta) * q2

        interpolated_quaternion = interpolated_quaternion / np.linalg.norm(interpolated_quaternion)
        interpolated_rotation = quat_to_R(interpolated_quaternion)

        return interpolated_rotation

    def interpolate_qsf(self, t):
        q1 = self.target_orientation_A
        q2 = self.target_orientation_B

        dot_product = np.dot(q1, q2)
        if dot_product < 0:
            q1 = -q1

        theta = np.arccos(dot_product)

        if theta == 0:
            interpolated_quaternion = (1 - t) * q1 + t * q2
        else:
            interpolated_quaternion = np.sin((1 - t) * theta) / np.sin(theta) * q1 + np.sin(t * theta) / np.sin(
                theta) * q2

        interpolated_quaternion = interpolated_quaternion / np.linalg.norm(interpolated_quaternion)
        interpolated_rotation = quat_to_R(interpolated_quaternion)

        return interpolated_rotation

    def render(self, t):

        # Draw labels
        theta = np.radians(90)  # 90-degree rotation around x axis so labels face viewer
        st = np.sin(theta)
        ct = np.cos(theta)

        # Rotation about x-axis for labels
        M = np.eye(4, dtype='f4')
        M[1][1] = ct
        M[1][2] = -st
        M[2][1] = st
        M[2][2] = ct
        M[0:3, 3] = self.pos
        M[0][3] -= 1  # center under monkey
        M[1][3] += -1.5  # place below monkey

        for c in self.rotorder:
            M[0][3] += 0.5  # place to the right of previous letter

            # Pass args to fragment shader (letters)
            self.prog['M'].write(M.T.flatten())
            self.prog["compute"] = False
            self.prog["objectColor"] = letter_colors[c]

            self.vao[c].render()

        # Translate monkey
        M = np.eye(4, dtype='f4')
        M[0:3, 3] = self.pos

        # Apply rotation
        if self.rotorder == 'RL':
            interpolated_rotation = self.interpolate_rl(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'QL':
            interpolated_rotation = self.interpolate_ql(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'QS':
            interpolated_rotation = self.interpolate_qs(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'QLN':
            interpolated_rotation = self.interpolate_qln(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'QLNF':
            interpolated_rotation = self.interpolate_qlnf(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'QSF':
            interpolated_rotation = self.interpolate_qsf(t)
            M[0:3, 0:3] = interpolated_rotation

        elif self.rotorder == 'A':
            rotation = quat_to_R(self.target_orientation_A)  # Need to convert to rotation matrix unlike other interpolations
            M[0:3, 0:3] = rotation

        elif self.rotorder == 'B':
            rotation = quat_to_R(self.target_orientation_B)
            M[0:3, 0:3] = rotation

        else:
            # Interpolate rotation, then set rotation
            M[0:3, 0:3] = self.interpolate_euler(t)

        self.prog['M'].write(M.T.flatten())  # transpose and flatten to get in Opengl Column-majfor format

        # Pass args to fragment shader (monkey)
        self.prog["lightPos"] = (10, 30, 20)
        self.prog["objectColor"] = (0.5, 0.5, 0.5)
        self.prog["compute"] = True

        self.vao['monkey'].render()


class HelloWorld(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Hello World"
    window_size = (1280, 720)
    aspect_ratio = 16.0 / 9.0
    resizable = True
    resource_dir = 'data'

    def setup_wire_box(self):
        # create cube vertices
        vertices = np.array([
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, 1.0, -1.0,
            -1.0, 1.0, -1.0,
            -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0,
            1.0, 1.0, 1.0,
            -1.0, 1.0, 1.0,
        ], dtype='f4')
        # create cube edges
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype='i4')
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo = self.ctx.buffer(indices.astype("i4").tobytes())
        # note that we can provide nothing to the normal attribute, as we will ignore it with the lighting disabled
        self.cube_vao = self.ctx.vertex_array(self.prog, [(vbo, '3f', 'in_position')], index_buffer=ibo, mode=mgl.LINES)

    def draw_wire_box(self):
        self.cube_vao.render()

    def draw_viewer(self):
        self.vao['monkey'].render()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random.seed(0)  # set random seed for deterministic reproducibility
        self.prog = self.ctx.program(
            vertex_shader=open('glsl/vert.glsl').read(),
            fragment_shader=open('glsl/frag.glsl').read())

        # load obj files for drawing the monkey and letters
        self.scene = {}
        self.vao = {}
        for a in ['monkey', 'X', 'Y', 'Z', 'R', 'L', 'N', 'Q', 'S', 'F', 'A', 'B']:
            self.scene[a] = self.load_scene(a + ".obj")
            self.vao[a] = self.scene[a].root_nodes[0].mesh.vao.instance(self.prog)
        self.setup_wire_box()

        # Set up a grid of bodies, nicely spaced for viewing
        self.bodies = []
        for i in range(len(rotation_type)):
            c = 4 * ((i % 5) - 2)
            r = 4 * (-(i // 5) + 1.75)
            self.bodies.append(Body(np.array([c, r, 0]), rotation_type[i], self.prog, self.vao))

        # initialize the target orientations
        self.A = np.array([1, 0, 0, 0])
        self.set_new_rotations(self.A, 0)  # Set initial orientation of all bodies to A
        self.B = np.array([1, 0, 0, 0])
        self.set_new_rotations(self.B, 1)  # Set initial orientation of all bodies to B

        # Setup the primary and secondary viewing and projection matrices        
        self.V1 = matrix44.create_look_at(eye=(0, 0, 40), target=(0, 0, 0), up=(0, 1, 0), dtype='f4')
        self.P1 = matrix44.create_perspective_projection(25.0, self.aspect_ratio, 10, 45, dtype='f4')
        self.V2 = matrix44.create_look_at(eye=(30, 10, 55), target=(0, 0, 20), up=(0, 1, 0), dtype='f4')
        self.P2 = matrix44.create_perspective_projection(40.0, self.aspect_ratio, 10, 100.0, dtype='f4')
        self.V_target = self.V1
        self.P_target = self.P1
        self.V_current = self.V1.copy()
        self.P_current = self.P1.copy()

    def set_new_rotations(self, target, i):
        for b in self.bodies: b.set_rotation(target, i)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.A:
                self.A = rand_unit_quaternion()
                self.set_new_rotations(self.A, 0)
            if key == self.wnd.keys.B:
                self.B = rand_unit_quaternion()
                self.set_new_rotations(self.B, 1)
            if key == self.wnd.keys.Z:
                q = quaternion_random_axis_angle(np.pi / 180 * 3)
                q = quaternion_multiply(q, self.A)
                self.B = -q
                self.set_new_rotations(self.B, 1)
            if key == self.wnd.keys.X:
                q = quaternion_random_axis_angle(np.pi)
                self.B = quaternion_multiply(q, self.A)
                self.set_new_rotations(self.B, 1)
            if key == self.wnd.keys.I:
                self.A = np.array([1, 0, 0, 0])
                self.B = np.array([1, 0, 0, 0])
                self.set_new_rotations(self.A, 0)
                self.set_new_rotations(self.B, 1)
            elif key == self.wnd.keys.NUMBER_1:
                self.V_target = self.V1  # eye=(0, 0, 40)
                self.P_target = self.P1
            elif key == self.wnd.keys.NUMBER_2:
                self.V_target = self.V2  # eye=(30, 10, 55)
                self.P_target = self.P2

    def render(self, time, frame_time):
        self.ctx.clear(0, 0, 0)
        self.ctx.enable(mgl.DEPTH_TEST)

        # Interpolate the current and target viewing and projection matrices
        self.V_current = self.V_current * 0.9 + self.V_target * 0.1
        self.P_current = self.P_current * 0.9 + self.P_target * 0.1
        self.prog['P'].write(self.P_current)
        self.prog['V'].write(self.V_current)

        time_mod_4 = time % 4
        if time_mod_4 < 1:
            t = time_mod_4
        elif time_mod_4 < 2:
            t = 1
        elif time_mod_4 < 3:
            t = 3 - time_mod_4
        else:
            t = 0

        for b in self.bodies:
            b.render(t)

        # Draw viewer
        M = np.linalg.inv(self.V1)
        M[2][2] *= -1  # negative rotation in z
        self.prog['M'].write(M)
        self.vao['monkey'].render()

        # Draw frustum
        M = np.linalg.inv(self.V1 @ self.P1)
        self.prog['M'].write(M)
        self.prog['compute'] = False
        self.draw_wire_box()


HelloWorld.run()
