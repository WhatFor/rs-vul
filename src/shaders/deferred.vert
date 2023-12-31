#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

layout(location = 0) out vec3 out_colour;
layout(location = 1) out vec3 out_normal;

layout(set = 0, binding = 0) uniform VPData {
    mat4 view;
    mat4 proj;
} vp;

layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normals;
} model;

void main() {
    gl_Position = vp.proj * vp.view * model.model * vec4(position, 1.0);
    out_colour = colour;
    out_normal = mat3(model.normals) * normal;
}