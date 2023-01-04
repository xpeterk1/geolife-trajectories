#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include "opengl/Map.h"
#include "model/Dataset.h"

#include "glm/gtx/string_cast.hpp"

#include "cuda/heatmap.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

std::unique_ptr<Map> map_ptr;
std::unique_ptr<Dataset> data_ptr;
const float zoom_sensitivity = 0.05f;
const float move_sensitivity = 0.02f;

int main() {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 800, "Geolife Trajectories", NULL, NULL);

	if (window == NULL) {
		std::cout << "Failed To Create Window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	map_ptr = std::make_unique<Map>();
	data_ptr = std::make_unique<Dataset>("data/data_small.txt",1000000, false);

	//std::ofstream file("data_small.txt");
	//for (const auto& point : data_ptr->data) 
	//{
	//	file << point.x << " " << point.y << " " << point.time << " " << point.mode << std::endl;
	//}

	//file.flush();
	//file.close();

	// Compute points of interest
	std::vector<float> heatmap = compute_heatmap(data_ptr.get()->data);

	int c = 0;
	for (float f : heatmap)
	{
		if (f != 0) c++;
	}

	int dim = pow(10, 4);
	unsigned int heatmap_texture;
	glGenTextures(1, &heatmap_texture);
	glBindTexture(GL_TEXTURE_2D, heatmap_texture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, dim, dim);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim, dim, GL_RED, GL_FLOAT, &heatmap[0]);
	glBindTexture(GL_TEXTURE_2D, 0);

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		map_ptr->Draw(heatmap_texture);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

//CALLBACKS
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double x, double y)
{

}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{

}