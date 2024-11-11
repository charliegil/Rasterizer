# COMP 557 Assignments

This repository contains my work for COMP 557 assignments, exploring core computer graphics techniques like rasterization, rotation interpolation, shading, and mesh subdivision. Below is an overview of each assignment, complete with videos and images showcasing the results.

---

## Assignment 1 - Rasterizer

In this assignment, I implemented a simple rasterizer in Python. The rasterizer processes 2D images by converting geometric descriptions of shapes into pixel data, simulating the basic steps of rendering.

### Rasterizer in Action

![Rasterizer Video](https://github.com/user-attachments/assets/4ac0f5a7-10ca-42e9-89e6-c96e32b68172)

---

## Assignment 2 - Rotation Interpolation and Modern GL Shading

This assignment focused on implementing various rotation interpolation methods, including **Euler angles** and **quaternions**. I also used Modern GL shaders to implement **Blinn-Phong shading**, which enhances the scene with realistic light reflections. Additionally, the system includes two eyepoint perspectives, enabled by appropriate coordinate-space transformations.

### Rotation Interpolation in Action

![Rotation Interpolation Video](https://github.com/user-attachments/assets/27d09b66-09d4-43ec-b736-831c817f7333)

### Alternate Eyepoint Perspective

![Alternate Eyepoint Perspective](https://github.com/user-attachments/assets/40710b12-d98a-45ec-b1e2-5e651623d377)

---

## Assignment 3 - Mesh Subdivision

In this assignment, I implemented a **mesh subdivision algorithm** using a **half-edge data structure (HEDS)**. The algorithm refines the mesh by subdividing its faces, and the interface allows visual exploration of mesh faces, vertices, and half-edges. Users can navigate between different subdivision levels, viewing the **odd and even vertices** used in the algorithm and the normals for each face.

### Subdivision Levels

- **Subdivision Level 0**

  ![Subdivision Level 0](https://github.com/user-attachments/assets/aa86cd6f-5b4f-4ac9-8467-efce701849fd)

- **Subdivision Level 1**

  ![Subdivision Level 1](https://github.com/user-attachments/assets/530a0283-99dd-46cb-95fa-d60d029c8bea)

- **Subdivision Level 2**

  ![Subdivision Level 2](https://github.com/user-attachments/assets/421566f1-e8e5-4de5-a439-02b32ec6bb1a)

---
