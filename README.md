# OpenGL K-Means Clustering Visualization

A 3D visualization of the K-means clustering algorithm using modern OpenGL. This application demonstrates both OpenGL rendering techniques and the K-means clustering algorithm in an interactive environment.

## Features

- **3D K-means Clustering**: Visualizes K-means clustering algorithm in 3D space
- **Interactive Camera**: Orbit camera with mouse controls for exploring the 3D space
- **Step-by-Step Visualization**: Press SPACE to advance the K-means algorithm one iteration at a time
- **Dynamic Point Rendering**: Points are rendered as smooth circles with colors based on cluster assignment
- **Coordinate Axes**: X, Y, and Z axes are displayed for spatial reference

## Controls

- **Mouse Drag (Left Button)**: Orbit the camera around the data points
- **Mouse Scroll**: Zoom in/out
- **SPACE**: Perform one iteration of the K-means algorithm
- **R**: Reset data points and centroids with new random positions
- **ESC**: Exit the application

## Technical Details

### OpenGL Features Used

- Modern OpenGL 3.3+ Core Profile
- Shader-based rendering pipeline
- Vertex Buffer Objects (VBOs) and Vertex Array Objects (VAOs)
- Dynamic buffer updates for real-time visualization
- Point sprites with fragment shader-based circle rendering

### Libraries

- **GLFW**: Window creation and input handling
- **GLAD**: OpenGL function loading
- **GLM**: Mathematics library for 3D transformations
- **Standard Library**: Random number generation, containers, and algorithms

### K-means Algorithm Implementation

The application implements the standard K-means clustering algorithm:

1. Initialize K centroids at random positions
2. Assign each data point to the nearest centroid
3. Recalculate centroids as the mean of all points assigned to them
4. Repeat steps 2-3 until convergence or maximum iterations reached

## Building the Project

### Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler
- OpenGL 3.3+ compatible graphics hardware

### Build Instructions

1. Clone the repository
2. Create a build directory:
   ```
   mkdir build
   cd build
   ```
3. Configure with CMake:
   ```
   cmake ..
   ```
4. Build the project:
   ```
   cmake --build .
   ```
5. Run the executable:
   ```
   ./OpenGLCube
   ```

## Project Structure

- `src/`: Source code files
  - `main.cpp`: Main application code and K-means implementation
  - `Shader.h`: Shader utility class
- `shaders/`: GLSL shader files
  - `point.vert/frag`: Shaders for rendering data points and centroids
  - `axis.vert/frag`: Shaders for rendering coordinate axes
- `external/`: External dependencies
  - `glad/`: OpenGL function loader
  - `glfw/`: Window and input management
  - `glm/`: Mathematics library

## License

This project is provided as an educational resource. MIT License.

Author: bigcodeneck