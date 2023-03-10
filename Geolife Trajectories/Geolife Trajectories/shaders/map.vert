#version 460 core

layout (location = 0) in vec3 aPos;
out vec2 FragCoord;

uniform mat4 modelMatrix;


void main(){
	FragCoord = (aPos * 0.5 + 0.5).xy;
	gl_Position = modelMatrix * vec4(aPos, 1.0);
};