#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "../opengl/Shader.h"

class Map
{
private:
	GLuint vao, vbo;
	Shader map_shader;

	const glm::vec3 square[4]
	{
		glm::vec3(1.0, 1.0, 0.0),
		glm::vec3(1.0, -1.0, 0.0),
		glm::vec3(-1.0, 1.0, 0.0),
		glm::vec3(-1.0, -1.0, 0.0)
	};

public:
	

private:


public:
	Map();
	~Map();
	void Draw();

};

