#version 460 core

in vec2 FragCoord;

uniform sampler2D map_texture;

out vec4 fragColor;
void main()
{
	vec4 color = texture(map_texture, FragCoord);
	fragColor = vec4(color.xyz, 1.0);
};