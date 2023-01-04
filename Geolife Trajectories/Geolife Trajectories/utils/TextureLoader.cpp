#include "TextureLoader.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

unsigned int TextureLoader::LoadTextureFromFile(char const* path)
{
	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	
	unsigned char* image_data = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	
	unsigned int textureID;
	glGenTextures(1, &textureID);

	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
	glGenerateMipmap(GL_TEXTURE_2D);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	stbi_image_free(image_data);

	glBindTexture(GL_TEXTURE_2D, 0);

	return textureID;
}
