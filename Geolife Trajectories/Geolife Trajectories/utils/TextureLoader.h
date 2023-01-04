#pragma once

#include<iostream>
#include<vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class TextureLoader {

public:
	unsigned int LoadTextureFromFile(char const* path);
};