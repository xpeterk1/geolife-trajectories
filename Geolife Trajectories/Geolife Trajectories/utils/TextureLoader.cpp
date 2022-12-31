#include "TextureLoader.h"

#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>

unsigned int TextureLoader::LoadTextureFromFile(char const* path, const unsigned int& textureUnit, GLint wrappingMethod, bool gammaCorrection)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	int Width, Height, nrOfComponents;
	stbi_set_flip_vertically_on_load(true);
	unsigned char* gameData = stbi_load(path, &Width, &Height, &nrOfComponents, 0);
	GLenum format;
	if (gameData)
	{
		if (nrOfComponents == 1) format = GL_RED;
		else if (nrOfComponents == 3) format = gammaCorrection ? GL_SRGB : GL_RGB;
		else if (nrOfComponents == 4) format = gammaCorrection ? GL_SRGB_ALPHA : GL_RGBA; //For png
		glActiveTexture(GL_TEXTURE0 + textureUnit);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, Width, Height, 0, format, GL_UNSIGNED_BYTE, gameData);
		glGenerateMipmap(GL_TEXTURE_2D);
		if (nrOfComponents == 4) // transparence requires settings to look right at the edges
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrappingMethod);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrappingMethod);
		}
		else
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		stbi_image_free(gameData);
		glActiveTexture(GL_TEXTURE0);
	}
	else
	{
		std::cout << "\nTexture failed to load at path: " << path << std::endl;
		stbi_image_free(gameData);
	}
	return textureID;
}
