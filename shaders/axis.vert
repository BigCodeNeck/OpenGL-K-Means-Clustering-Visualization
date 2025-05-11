#version 330 core

// Input vertex attribute
layout (location = 0) in vec3 aPos;  // Position attribute (x, y, z coordinates)

// Combined transformation matrix
uniform mat4 MVP;  // Combined Model-View-Projection matrix for efficiency

void main() {
    // Transform vertex position directly with combined MVP matrix
    // This is more efficient than applying model, view, and projection separately
    // Used for simple objects like coordinate axes where we don't need intermediate results
    gl_Position = MVP * vec4(aPos, 1.0);
}