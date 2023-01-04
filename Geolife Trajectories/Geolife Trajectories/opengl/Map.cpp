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

    scale_factor = 0.8f;
    translation = glm::vec2(0.0f);
    Transform();

    texture_id = TextureLoader().LoadTextureFromFile("resources\\map.png");
}

Map::~Map()
{
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
}

void Map::Draw()
{
    map_shader.use();
    map_shader.setInt("map_texture", 0);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Map::AddScale(float scale_factor) 
{
    this->scale_factor += scale_factor;
    Transform();
}

void Map::Translate(glm::vec2 direction)
{
    this->translation += direction;
    Transform();
}

void Map::Transform() 
{
    map_shader.use();
    glm::mat4 modelMat = glm::scale(glm::translate(glm::mat4(1.0), glm::vec3(this->translation, 0.0f)), glm::vec3(scale_factor));
    map_shader.setMat4F("modelMatrix", modelMat);
}