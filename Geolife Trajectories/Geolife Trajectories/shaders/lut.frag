#version 460 core

in vec2 FragCoord;
out vec4 fragColor;

void main()
{

	fragColor = vec4(FragCoord.xy, 0.0, 1.0);
};