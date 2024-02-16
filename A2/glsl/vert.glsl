// Charles Gil, 260970950
#version 330 core

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

out vec3 FragPos;
out vec3 Normal;
out vec3 EyePos;

void main() {
	// Extract the upper-left 3x3 submatrix of the viewing matrix
    mat3 viewMatrix3x3 = mat3(V);

    // Calculate the eye position in world space
    EyePos = -inverse(viewMatrix3x3) * V[3].xyz;

	FragPos = (V * M * vec4(in_position, 1.0)).xyz; // Unsure about what matrices to apply
	gl_Position = P * V * M * vec4(in_position, 1.0); // M: Object space to world space, V: World space to camera space, P: View space to clip space
	mat3 normalMatrix = transpose(inverse(mat3(M)));
    Normal = normalize(normalMatrix * in_normal);
}