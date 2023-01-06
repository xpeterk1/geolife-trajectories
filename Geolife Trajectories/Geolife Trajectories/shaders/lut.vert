#version 460 core

layout (location = 0) in vec3 aPos;
out vec2 FragCoord;

void main(){
	FragCoord = (aPos * 0.5 + 0.5).xy;
	gl_Position = vec4(aPos, 1.0);
};