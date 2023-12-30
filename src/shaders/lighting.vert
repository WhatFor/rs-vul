#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 frag_position;

layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

void main() {
    mat4 worldview = ubo.view * ubo.model;
    gl_Position = ubo.projection * worldview * vec4(position, 1.0);
    frag_position = vec3(ubo.model * vec4(position, 1.0));
}