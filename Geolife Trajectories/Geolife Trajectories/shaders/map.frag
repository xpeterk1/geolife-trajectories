#version 460 core

in vec2 FragCoord;

uniform sampler2D map_texture;
uniform sampler2D heatmap_texture;

out vec4 fragColor;
void main()
{
	vec4 map_color = texture(map_texture, FragCoord);
	vec4 heatmap_color = texture(heatmap_texture, FragCoord);
	
	vec3 color;
	if (heatmap_color.r != 0)
		color = heatmap_color.xyz;
	else 
		color = map_color.xyz;

	fragColor = vec4(color, 1.0);
};