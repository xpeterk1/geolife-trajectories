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
}

Map::~Map()
{
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
}

void Map::Draw()
{
    map_shader.use();
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}