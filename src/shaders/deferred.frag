// #version 460

// layout(location = 0) in vec3 in_colour;
// layout(location = 1) in vec3 in_normal;
// layout(location = 2) in vec3 in_frag_position;

// layout(location = 0) out vec4 f_colour;

// layout(set = 0, binding = 1) uniform AmbientLight {
//     vec3 colour;
//     float intensity;
// } ambient_light;

// layout(set = 0, binding = 2) uniform DirectionalLight {
//     vec3 colour;
//     vec3 position;
// } dir_light;

// void main() {
//     // ambient
//     vec3 ambient_colour = ambient_light.intensity * ambient_light.colour;

//     // point, diffuse
//     vec3 light_direction = normalize(dir_light.position - in_frag_position);
//     float diffuse_intensity = max(dot(in_normal, light_direction), 0.0);
//     vec3 diffuse_colour = diffuse_intensity * dir_light.colour;

//     vec3 combined_colour = (ambient_colour + diffuse_colour) * in_colour;

//     f_colour = vec4(combined_colour, 1.0);
// }

#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

void main() {
    f_color = vec4(in_color, 1.0);
    f_normal = in_normal;
}