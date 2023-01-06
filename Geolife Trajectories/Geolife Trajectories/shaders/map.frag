#version 460 core

in vec2 FragCoord;

uniform sampler2D map_texture;
uniform sampler2D heatmap_texture;
uniform sampler2D lut_texture;

out vec4 fragColor;
void main()
{
	vec4 map_color = texture(map_texture, FragCoord);
	vec4 heatmap_color = texture(heatmap_texture, FragCoord);
	vec4 lut_color = texture(lut_texture, vec2(0.0, 1.0 - heatmap_color.r));

	vec3 color;
	if (heatmap_color.r != 0)
		color = 0.1 * map_color.xyz + 0.9 * lut_color.xyz;
	else 
		color = map_color.xyz;

	fragColor = vec4(color, 1.0);
};