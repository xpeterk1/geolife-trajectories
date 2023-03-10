#include "Map.h"

Map::Map() : map_shader("shaders\\map.vert", "shaders\\map.frag")
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec3), &square[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    scale_factor = 1.0f;
    translation = glm::vec2(0.0f);

    texture_id = TextureLoader().LoadTextureFromFile("resources\\map.png");
}

Map::~Map()
{
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
    glDeleteTextures(1, &texture_id);
}

void Map::Draw(unsigned int heatmap_texture_id, unsigned int lut_texture)
{
    map_shader.use();
    glm::mat4 modelMat = glm::scale(glm::translate(glm::mat4(1.0), glm::vec3(this->translation, 0.0f)), glm::vec3(scale_factor));
    map_shader.setMat4F("modelMatrix", modelMat);
    map_shader.setInt("map_texture", 0);
    map_shader.setInt("heatmap_texture", 1);
    map_shader.setInt("lut_texture", 2);
    
    glBindVertexArray(vao);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, heatmap_texture_id);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, lut_texture);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    glBindTexture(GL_TEXTURE_2D, 0);

    if (this->targetTranslation != this->translation) {
        auto dir = this->targetTranslation - this->translation;

        this->translation += dir * 0.1f;
    }
}

void Map::AddScale(float scale_factor) 
{
    this->scale_factor += scale_factor;

    if (this->scale_factor < 1) this->scale_factor = 1;
}

void Map::Translate(glm::vec2 direction)
{    
    this->targetTranslation = this->translation + direction;
}

void Map::Reset() 
{
    scale_factor = 1;
    targetTranslation = glm::vec2(0.0f);
}