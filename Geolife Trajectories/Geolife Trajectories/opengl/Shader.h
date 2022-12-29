#pragma once

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

class Shader
{

private:
	// The shader program Id
	unsigned int ID;

public:

	// constructor reads and builds the shader
	Shader(const char* vertexShaderPath, const char* fragmentShaderPath);

	// descructor
	~Shader();

	// use/activate the shader
	void use();

	// utility uniform functions to manipulate uniforms within the shader program
	void setBool(const std::string& name, const bool& value) const;
	void setInt(const std::string& name, const int value) const;
	void setUInt(const std::string& name, const unsigned int value) const;
	void setFloat(const std::string& name, const float& value) const;
	void setVec3F(const std::string& name, const float& value1, const float& value2, const float& value3) const;
	void setVec3F(const std::string& name, const glm::vec3& vector) const;
	void setVec2F(const std::string& name, const glm::vec2& vector) const;
	void setMat4F(const std::string& name, const glm::mat4& matrix) const;
	float getFloat(const std::string& name);

private:
	int compileAndCheckShader(int shaderId);
};