#pragma once

#include<iostream>
#include<vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class TextureLoader {

public:
	unsigned int LoadTextureFromFile(char const* path, const unsigned int& textureUnit = 0, GLint wrappingMethod = GL_CLAMP_TO_EDGE, bool gammaCorrection = false);
};