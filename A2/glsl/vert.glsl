#version 330 core

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

in vec3 in_position;

void main() {	
	gl_Position = P * V * M * vec4(in_position, 1.0);
}