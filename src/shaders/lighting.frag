#version 450

layout(location = 0) in vec3 frag_pos;

// input_attachment_index, depending on order given in renderpass, not descriptor set
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(set = 0, binding = 3) uniform AmbientLight {
    vec3 colour;
    float intensity;
} ambient;

layout(set = 0, binding = 4) uniform DirectionalLight {
    vec4 position;
    vec3 colour;
} directional;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.colour;
    vec3 light_direction = normalize(directional.position.xyz - frag_pos);
    float directional_intensity = max(dot(normalize(subpassLoad(u_normals).rgb), light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.colour;
    vec3 combined_color = (ambient_color + directional_color) * subpassLoad(u_color).rgb;
    //f_color = vec4(combined_color, 1.0);
    f_color = vec4(subpassLoad(u_normals).rgb, 1.0);
}