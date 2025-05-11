#version 330 core

// Input vertex attributes
layout (location = 0) in vec3 aPos;    // Position attribute (x, y, z coordinates)
layout (location = 1) in vec3 aColor;  // Color attribute (r, g, b values)

// Output to fragment shader
out vec3 vs_color;  // Pass color to fragment shader

// Transformation matrices
uniform mat4 model;      // Model matrix (object space to world space)
uniform mat4 view;       // View matrix (world space to camera space)
uniform mat4 projection; // Projection matrix (camera space to clip space)
uniform float u_pointSize; // Size of the point in pixels

void main() {
    // Apply MVP transformation to get final position in clip space
    // Order: model (object → world) → view (world → camera) → projection (camera → clip)
    gl_Position = projection * view * model * vec4(aPos, 1.0);

    // Pass color to fragment shader without modification
    vs_color = aColor;

    // Set the size of the point (in pixels)
    // This will be used by the fragment shader to create circular points
    gl_PointSize = u_pointSize;
}