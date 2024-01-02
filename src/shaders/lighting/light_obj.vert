#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform VPData {
    mat4 view;
    mat4 projection;
} vp_uniforms;

layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normals;
} model;

void main() {
    gl_Position = vp_uniforms.projection * vp_uniforms.view * model.model * vec4(position, 1.0);
    out_color = color;
}