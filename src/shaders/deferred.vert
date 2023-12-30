#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

layout(location = 0) out vec3 out_colour;
layout(location = 1) out vec3 out_normal;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    mat4 worldview = ubo.view * ubo.model;

    gl_Position = ubo.proj * worldview * vec4(position, 1.0);
    out_colour = colour;
    out_normal = mat3(ubo.model) * normal;
}