#version 330 core

// Output color
out vec4 FragColor;  // Final output color for this fragment

// Uniform variable for axis color
uniform vec3 lineColor;  // Color of the axis line (set from application)

void main() {
    // Set the final color with full opacity (alpha = 1.0)
    // This is a simple fragment shader that just outputs the color passed from the application
    // Used for coordinate axes where each axis has a different color (typically R=X, G=Y, B=Z)
    FragColor = vec4(lineColor, 1.0);
}