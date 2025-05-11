#version 330 core

// Output color
out vec4 FragColor;  // Final output color for this fragment

// Input from vertex shader
in vec3 vs_color;    // Color passed from vertex shader

void main() {
    // Calculate distance from center of point
    // gl_PointCoord is a built-in variable that contains the coordinate of the fragment within the point
    // It ranges from (0,0) to (1,1), so we subtract 0.5 to center it at origin
    vec2 circCoord = gl_PointCoord - vec2(0.5);

    // Discard fragments outside the circle with radius 0.5
    // This creates circular points instead of squares
    if (length(circCoord) > 0.5) {
        discard;  // Don't render this fragment
    }

    // Set the final color with full opacity (alpha = 1.0)
    FragColor = vec4(vs_color, 1.0);
}