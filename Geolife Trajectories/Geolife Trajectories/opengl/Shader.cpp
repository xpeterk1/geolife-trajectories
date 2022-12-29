#include "Shader.h"

//compile shader and check for compile-time errors of thevertex shaders' source code compilation
int Shader::compileAndCheckShader(int shaderId)
{
	//Compile Shader referred to by shaderId
	glCompileShader(shaderId);
	int success;
	char infoLog[1024];
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &success);

	//Read Log from vertex shader compilation and output it if error occured
	if (!success)
	{
		glGetShaderInfoLog(shaderId, 1024, NULL, infoLog);
		std::cout << "\nERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	return success;
}

Shader::Shader(const char* vertexShaderPath, const char* fragmentShaderPath)
{
	// 1. retrieve the vertex/fragment source code from filePath //
	//-----------------------------------------------------------//

	// Define file streams for reading shader code from hdd
	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;

	// Define strings to store shader code
	std::string vertexShaderCodeString;
	std::string fragmentShaderCodeString;

	// ensure ifstream objects can throw exceptions:
	vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		// open files from file path
		try
		{
			vertexShaderFile.open(vertexShaderPath);
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "\nERROR: Could not read Vertex Shader File: " << vertexShaderPath;
		}
		try
		{
			fragmentShaderFile.open(fragmentShaderPath);
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "\nERROR: Could not read Fragment Shader File: " << fragmentShaderPath;
		}

		// read file's buffer contents into streams
		std::stringstream vShaderStream, fShaderStream;
		vShaderStream << vertexShaderFile.rdbuf();
		fShaderStream << fragmentShaderFile.rdbuf();

		// close file handlers
		try
		{
			vertexShaderFile.close();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "\nERROR: Could not close Vertex Shader File";
		}
		try
		{
			fragmentShaderFile.close();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "\nERROR: Could not close Fragment Shader File";
		}

		// convert stream into string
		vertexShaderCodeString = vShaderStream.str();
		fragmentShaderCodeString = fShaderStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "\nERROR: Could not read or close Shader Files";
	}

	// 2. Compile shaders
	//--------------------//
	// Create a vertex shader and a fragment shader and save its IDs
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Attach shader source code to each shader object
	// Args: shader object, number of strings passed as source code, source code, ?
	const char* vertexShaderCode = vertexShaderCodeString.c_str();
	const char* fragmentShaderCode = fragmentShaderCodeString.c_str();
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(vertexShader);

	// Compile each shader and check if compilation errors occured (writes errors to console)
	compileAndCheckShader(vertexShader);
	compileAndCheckShader(fragmentShader);

	// Create a shader Program Object
	ID = glCreateProgram();

	// Link the vertex shader and the Fragment shader by first attaching them to "shaderProgram" (in the right order) 
	// and then linking them inside "shaderProgram"
	glAttachShader(ID, vertexShader);
	glAttachShader(ID, fragmentShader);
	glLinkProgram(ID);

	// Check for Linking Errors of "shaderProgram"
	int success;
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success)
	{
		GLchar infoLog[1024];
		glGetProgramInfoLog(ID, 1024, NULL, infoLog);
		std::cout << "\nERROR::SHADER::PROGRAM::LINK_FAILED\n" << infoLog << std::endl;
	}

	// Delete shader objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

Shader::~Shader()
{
	glDeleteProgram(ID);
}

void Shader::use()
{
	glUseProgram(ID);
}

void Shader::setBool(const std::string& name, const bool& value) const
{
	// query adress of uniform with provided name in the shader program
	int uniformLocation = glGetUniformLocation(ID, name.c_str());

	// set the value of the uniform at the adress we just queried to the current greenvalue
	glUniform1i(uniformLocation, value);
};

void Shader::setInt(const std::string& name, const int value) const
{
	// query address of the uniform with provided name nd the shader program (with id = ID)
	int uniformLocation = glGetUniformLocation(ID, name.c_str());

	// set the value of the uniform at the queried location
	glUniform1i(uniformLocation, value);
};

void Shader::setUInt(const std::string& name, const unsigned int value) const
{
	// query address of the uniform with provided name nd the shader program (with id = ID)
	int uniformLocation = glGetUniformLocation(ID, name.c_str());

	// set the value of the uniform at the queried location
	glUniform1ui(uniformLocation, value);
};

void Shader::setFloat(const std::string& name, const float& value) const
{
	//see other setters for syntax
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
};

void Shader::setVec3F(const std::string& name, const float& value1, const float& value2, const float& value3) const
{
	glUniform3f(glGetUniformLocation(ID, name.c_str()), value1, value2, value3);
}

void Shader::setVec3F(const std::string& name, const glm::vec3& vector) const
{
	glUniform3f(glGetUniformLocation(ID, name.c_str()), vector.x, vector.y, vector.z);
}

void Shader::setVec2F(const std::string& name, const glm::vec2& vector) const
{
	glUniform2f(glGetUniformLocation(ID, name.c_str()), vector.x, vector.y);
}

void Shader::setMat4F(const std::string& name, const glm::mat4& matrix) const
{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &matrix[0][0]);
}

float Shader::getFloat(const std::string& name)
{
	float retVal = 0;
	glGetUniformfv(this->ID, glGetUniformLocation(this->ID, name.c_str()), &retVal);
	return retVal;
}
