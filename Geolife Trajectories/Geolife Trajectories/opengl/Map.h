#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Shader.h"
#include "../utils/TextureLoader.h"

class Map
{
private:
	GLuint vao, vbo, texture_id;
	Shader map_shader;
	float scale_factor = 1;
	glm::vec2 translation;
	glm::vec2 targetTranslation;

	const glm::vec3 square[4]
	{
		glm::vec3(1.0, 1.0, 0.0),
		glm::vec3(1.0, -1.0, 0.0),
		glm::vec3(-1.0, 1.0, 0.0),
		glm::vec3(-1.0, -1.0, 0.0)
	};

public:
	

private:
	void Transform();

public:
	Map();
	~Map();
	void AddScale(float scale_factor);
	void Translate(glm::vec2 direction);
	void Draw(unsigned int heatmap_texture);
	void Reset();

};

