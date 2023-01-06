#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

#include "opengl/Map.h"
#include "model/HeatmapConfig.h"
#include "model/Dataset.h"
#include "cuda/heatmap.h"

#include "glm/gtx/string_cast.hpp"

#include <stdio.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void DrawGUI();

std::unique_ptr<Map> map_ptr;
std::unique_ptr<Dataset> data_ptr;
const float zoom_sensitivity = 0.1f;
const float move_sensitivity = 0.075f;
GLuint lut_texture;

HeatmapConfig heatmap_config;

int main() {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

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
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	map_ptr = std::make_unique<Map>();
	data_ptr = std::make_unique<Dataset>("data/data.bin", true);
	
	//FILE* fp;
	//fopen_s(&fp, "data.bin", "wb");
	//size_t items_written = fwrite(data_ptr->data.data(), sizeof(Datapoint), data_ptr->size, fp);
	//fclose(fp);

	// Initialize and upload data to cuda
	init_heatmap_data(data_ptr.get()->data);

	// Prepare output texture for heatmap
	// MUST BE THE SAME POWER AS PRECISION IN HEATMAP.CU
	int dim = pow(10, 3);
	GLuint heatmap_texture;
	glGenTextures(1, &heatmap_texture);
	glBindTexture(GL_TEXTURE_2D, heatmap_texture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, dim, dim);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Load lookup table texture
	lut_texture = TextureLoader().LoadTextureFromFile("resources/lut.png");

	//Init ImGUI
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//New ImGUI Frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Draw GUI
		DrawGUI();

		// Some GUI element changed
		if (heatmap_config.NeedsRecomputation())
		{
			std::vector<float> heatmap = compute_heatmap(heatmap_config);

			glBindTexture(GL_TEXTURE_2D, heatmap_texture);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim, dim, GL_RED, GL_FLOAT, &heatmap[0]);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		// Draw map with heatmap and Lookup table
		map_ptr->Draw(heatmap_texture, lut_texture);

		// Render GUI
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	//Destroy ImGUI context
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	//Free cuda related buffers
	free_heatmap_data();

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

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	map_ptr->AddScale(yoffset * zoom_sensitivity);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_LEFT)
		map_ptr->Translate(glm::vec2(move_sensitivity,0));
	if (key == GLFW_KEY_RIGHT)
		map_ptr->Translate(glm::vec2(-move_sensitivity, 0));
	if (key == GLFW_KEY_UP)
		map_ptr->Translate(glm::vec2(0, -move_sensitivity));
	if (key == GLFW_KEY_DOWN)
		map_ptr->Translate(glm::vec2(0, move_sensitivity));
}

void DrawGUI() 
{
	ImGui::Begin("Main Menu");

	// Checkboxes for transportation modes
	ImGui::Text("Transportation mode");

	if (ImGui::Checkbox("Unknown", &heatmap_config.unknown))
		heatmap_config.Switch(UNKNOWN);

	if (ImGui::Checkbox("Walk", &heatmap_config.walk))
		heatmap_config.Switch(WALK);

	if (ImGui::Checkbox("Bike", &heatmap_config.bike))
		heatmap_config.Switch(BIKE);

	if (ImGui::Checkbox("Bus", &heatmap_config.bus))
		heatmap_config.Switch(BUS);
	
	if (ImGui::Checkbox("Car", &heatmap_config.car))
		heatmap_config.Switch(CAR);
	
	if (ImGui::Checkbox("Subway", &heatmap_config.subway))
		heatmap_config.Switch(SUBWAY);

	if (ImGui::Checkbox("Train", &heatmap_config.train))
		heatmap_config.Switch(TRAIN);

	if (ImGui::Checkbox("Airplane", &heatmap_config.airplane))
		heatmap_config.Switch(AIRPLANE);
	
	if (ImGui::Checkbox("Boat", &heatmap_config.boat))
		heatmap_config.Switch(BOAT);

	if (ImGui::Checkbox("Run", &heatmap_config.run))
		heatmap_config.Switch(RUN);

	if (ImGui::Checkbox("Motorcycle", &heatmap_config.motorcycle))
		heatmap_config.Switch(MOTORCYCLE);

	static float begin = 10, end = 90;
	ImGui::DragFloatRange2("Time", &begin, &end, 0.25f, 0.0f, 100.0f, "Min: %.1f %%", "Max: %.1f %%");

	ImGui::Checkbox("Use Log Scale", &heatmap_config.use_log_scale);

	if (ImGui::Button("Reset Map"))
		map_ptr->Reset();

	ImGui::End();

	// LOOKUP TABLE
	bool open = true;
	ImGui::Begin("LUT", &open, ImGuiWindowFlags_NoTitleBar);
	auto windowWidth = ImGui::GetWindowSize().x;
	auto textWidth = ImGui::CalcTextSize("0").x;

	ImGui::SetCursorPosX((windowWidth - textWidth) * 0.5f);
	ImGui::BeginGroup();
	ImGui::Text("1");
	ImGui::Image((void*)(intptr_t)lut_texture, ImVec2(10, 200));
	ImGui::Text("0");
	ImGui::EndGroup();
	ImGui::End();
}