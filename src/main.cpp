/**
 * OpenGL K-Means Clustering Visualization
 *
 * This application visualizes the K-means clustering algorithm in 3D space using modern OpenGL.
 * It demonstrates both OpenGL rendering techniques and the K-means algorithm implementation.
 *
 * Features:
 * - 3D visualization of data points and cluster centroids
 * - Interactive camera controls (orbit and zoom)
 * - Step-by-step K-means algorithm visualization
 * - Dynamic point coloring based on cluster assignment
 *
 * Controls:
 * - Mouse Drag (Left Button): Orbit camera
 * - Mouse Scroll: Zoom in/out
 * - SPACE: Perform one iteration of K-means
 * - R: Reset data with new random positions
 * - ESC: Exit application
 *
 * Author: bigcodeneck. Please subscribe to my channel for more content: https://www.youtube.com/@bigcodeneck
 *
 */

// --- OpenGL and Window Management Libraries ---
#include <glad/glad.h> // OpenGL function loader (must be included first)
#include <GLFW/glfw3.h> // Window creation and input handling

// --- Mathematics Libraries ---
#include <glm/glm.hpp>                  // GLM core (vectors, matrices)
#include <glm/gtc/matrix_transform.hpp> // Matrix transformations (perspective, lookAt, etc.)
#include <glm/gtc/type_ptr.hpp>         // Value pointer access for OpenGL functions

// --- Custom Classes ---
#include "Shader.h" // Shader loading and management class

// --- Standard Library ---
#include <iostream>  // Console I/O
#include <vector>    // Dynamic arrays
#include <string>    // String handling
#include <cmath>     // Mathematical functions
#include <algorithm> // Algorithms (min, max, etc.)
#include <random>    // Random number generation
#include <limits>    // Numeric limits

// Define PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Application Configuration ---
const int NUM_POINTS = 100;             // Number of data points to generate
const int K_CLUSTERS = 3;               // Number of clusters for K-means algorithm
const int MAX_ITERATIONS = 30;          // Maximum iterations for K-means algorithm
const float POINT_SIZE_DATA = 5.0f;     // Size of data points in pixels
const float POINT_SIZE_CENTROID = 10.0f; // Size of centroid points in pixels (larger for visibility)
const float DATA_RANGE_MIN = 0.0f;      // Minimum coordinate value for data points
const float DATA_RANGE_MAX = 100.0f;    // Maximum coordinate value for data points
int WINDOW_WIDTH = 1024;                // Initial window width in pixels
int WINDOW_HEIGHT = 768;                // Initial window height in pixels

// --- Camera Configuration ---
glm::vec3 cameraPos;                    // Camera position (calculated from target, distance, yaw, pitch)
glm::vec3 cameraTarget = glm::vec3(DATA_RANGE_MAX / 2.0f, DATA_RANGE_MAX / 2.0f, DATA_RANGE_MAX / 2.0f); // Look-at target (center of data)
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f); // Up vector for camera orientation
float cameraDistance = 200.0f;          // Distance from camera to target
float cameraYaw = -135.0f;              // Horizontal rotation angle (in degrees)
float cameraPitch = -30.0f;             // Vertical rotation angle (in degrees)

// --- Mouse Input State ---
double lastMouseX = WINDOW_WIDTH / 2.0; // Last mouse X position for calculating movement delta
double lastMouseY = WINDOW_HEIGHT / 2.0; // Last mouse Y position for calculating movement delta
bool firstMouse = true;                 // Flag to handle first mouse input
bool leftMouseButtonPressed = false;    // Track if left mouse button is currently pressed
float mouseSensitivity = 0.1f;          // Mouse movement sensitivity for camera rotation
float scrollSensitivity = 1.0f;         // Scroll wheel sensitivity for camera zoom

// --- Data Structures ---
// Represents a 3D data point with position and cluster assignment
struct Point3D {
    glm::vec3 pos;      // 3D position (x, y, z)
    int clusterId;      // ID of the cluster this point belongs to (-1 = unassigned)

    // Constructor with default values
    Point3D(glm::vec3 _pos = glm::vec3(0.0f), int _id = -1) : pos(_pos), clusterId(_id) {}
};

// --- Data Containers ---
std::vector<Point3D> dataPoints;         // Collection of all data points
std::vector<Point3D> centroids;          // Collection of cluster centroids
std::vector<glm::vec3> clusterColorsVec; // Colors for each cluster (RGB values)

// --- OpenGL Object IDs ---
GLuint dataPointsVAO, dataPointsVBO;     // Vertex Array/Buffer Objects for data points
GLuint centroidsVAO, centroidsVBO;       // Vertex Array/Buffer Objects for centroids
GLuint axisVAO, axisVBO;                 // Vertex Array/Buffer Objects for coordinate axes

// --- Vertex Data Buffers ---
std::vector<float> pointVertexData;      // Buffer for data points (format: x,y,z, r,g,b)
std::vector<float> centroidVertexData;   // Buffer for centroids (format: x,y,z, r,g,b)

// --- K-means Algorithm State ---
int currentIteration = 0;                // Current iteration of the K-means algorithm
bool converged = false;                  // Flag indicating if K-means has converged

// --- Random Number Generation ---
std::mt19937 rng(std::random_device{}());                        // Mersenne Twister random number generator
std::uniform_real_distribution<float> dist_coord(DATA_RANGE_MIN, DATA_RANGE_MAX); // Distribution for random coordinates
std::uniform_int_distribution<int> dist_idx(0, NUM_POINTS - 1);  // Distribution for random indices (adjusted if NUM_POINTS changes)

// --- Helper & K-means Algorithm Functions ---

//Calculate squared distance between two 3D points

float distanceSq3D(const glm::vec3& p1, const glm::vec3& p2) {
    float dist = glm::distance(p1, p2);
    return dist * dist;
}

// Generate random data points in 3D space
void generateRandomData3D() {
    // Clear existing data
    dataPoints.clear();

    // Update random distribution if needed
    if (NUM_POINTS > 0) {
        dist_idx = std::uniform_int_distribution<int>(0, NUM_POINTS - 1);
    }

    // Generate random points
    for (int i = 0; i < NUM_POINTS; ++i) {
        dataPoints.emplace_back(glm::vec3(
            dist_coord(rng),  // Random X coordinate
            dist_coord(rng),  // Random Y coordinate
            dist_coord(rng)   // Random Z coordinate
        ));
    }
}

// Initialize K cluster centroids
void initializeCentroids3D() {
    centroids.clear();

    // Handle edge case: no data points but clusters requested
    if (NUM_POINTS == 0 && K_CLUSTERS > 0) {
        std::cerr << "Warning: NUM_POINTS is 0. Cannot initialize centroids from data points." << std::endl;
        // Create random centroids instead
        for (int i = 0; i < K_CLUSTERS; ++i) {
            centroids.emplace_back(glm::vec3(dist_coord(rng), dist_coord(rng), dist_coord(rng)));
        }
        return;
    }

    // No clusters requested
    if (K_CLUSTERS == 0) return;

    // Track which data points have been used as centroids
    std::vector<int> used_indices;

    // Initialize each centroid
    for (int i = 0; i < K_CLUSTERS; ++i) {
        // Safety check
        if (NUM_POINTS == 0) break;

        // If we've used all available points but need more centroids
        if (used_indices.size() >= NUM_POINTS && i < K_CLUSTERS) {
            // Create random centroids instead
            centroids.emplace_back(glm::vec3(dist_coord(rng), dist_coord(rng), dist_coord(rng)));
            continue;
        }

        // Try to find a unique data point to use as centroid
        int randomIndex;
        bool unique;
        int attempts = 0;

        // Keep trying until we find a unique index or give up after too many attempts
        do {
            unique = true;
            randomIndex = dist_idx(rng);

            // Check if this index has already been used
            for (int used_idx : used_indices) {
                if (used_idx == randomIndex) {
                    unique = false;
                    break;
                }
            }
            attempts++;
        } while (!unique && attempts < NUM_POINTS * 2);

        // Add the selected point as a centroid
        if (unique) {
            centroids.emplace_back(dataPoints[randomIndex].pos);
            used_indices.push_back(randomIndex);
        } else {
            // Fallback: just use any random point if we couldn't find a unique one
            centroids.emplace_back(dataPoints[dist_idx(rng)].pos);
        }
    }
}

// Generate distinct colors for each cluster
void generateClusterColors() {
    clusterColorsVec.clear();

    // Use fixed seed for reproducible colors
    std::mt19937 color_rng(12345);
    std::uniform_real_distribution<float> color_dist(0.2f, 0.9f);

    // Generate a color for each cluster
    for (int i = 0; i < K_CLUSTERS; ++i) {
        clusterColorsVec.push_back(glm::vec3(
            color_dist(color_rng),  // Random red component
            color_dist(color_rng),  // Random green component
            color_dist(color_rng)   // Random blue component
        ));
    }

    // Add gray color for unassigned points
    clusterColorsVec.push_back(glm::vec3(0.5f, 0.5f, 0.5f));
}

// Assign each data point to its nearest centroid
bool assignPointsToClusters3D() {
    // Safety check
    if (K_CLUSTERS == 0 || centroids.empty()) return false;

    bool changed = false;

    // Process each data point
    for (auto& point : dataPoints) {
        float minDistSq = std::numeric_limits<float>::max();
        int newClusterId = -1;

        // Find the closest centroid
        for (int k = 0; k < K_CLUSTERS; ++k) {
            if (k >= centroids.size()) continue;

            // Calculate squared distance to this centroid
            float dSq = distanceSq3D(point.pos, centroids[k].pos);

            // If this is the closest centroid so far, update
            if (dSq < minDistSq) {
                minDistSq = dSq;
                newClusterId = k;
            }
        }

        // If the point's cluster assignment changed, update it
        if (point.clusterId != newClusterId) {
            point.clusterId = newClusterId;
            changed = true;
        }
    }

    return changed;
}

// Update centroid positions based on assigned points
bool updateCentroids3D() {
    // Safety check
    if (K_CLUSTERS == 0 || dataPoints.empty()) return false;

    // Initialize accumulators for new centroid positions and counts
    std::vector<glm::vec3> newCentroidPositions(K_CLUSTERS, glm::vec3(0.0f));
    std::vector<int> counts(K_CLUSTERS, 0);
    bool changed = false;

    // Sum up positions of all points in each cluster
    for (const auto& point : dataPoints) {
        if (point.clusterId != -1 && point.clusterId < K_CLUSTERS) {
            newCentroidPositions[point.clusterId] += point.pos;
            counts[point.clusterId]++;
        }
    }

    // Calculate new centroid positions as mean of points in cluster
    for (int k = 0; k < K_CLUSTERS; ++k) {
        if (k >= centroids.size()) continue; // Safety check

        // Only update if cluster has points
        if (counts[k] > 0) {
            // Calculate mean position
            glm::vec3 newPos = newCentroidPositions[k] / (float)counts[k];

            // Check if position changed significantly
            if (distanceSq3D(centroids[k].pos, newPos) > 1e-6f) {
                changed = true;
            }

            // Update centroid position
            centroids[k].pos = newPos;
        }
        // Note: Empty clusters are left at their current position
    }

    return changed;
}

// Perform one iteration of the K-means algorithm
void stepKMeans() {
    // Check if algorithm has already finished
    if (converged || currentIteration >= MAX_ITERATIONS) {
        std::cout << "KMeans: Converged or max iterations reached." << std::endl;
        return;
    }

    // Log current iteration
    std::cout << "Iteration: " << currentIteration + 1;

    // Execute K-means steps
    bool assignmentsChanged = assignPointsToClusters3D();
    bool centroidsMoved = updateCentroids3D();

    // Check for convergence (no changes in assignments or positions)
    if (!assignmentsChanged && !centroidsMoved) {
        converged = true;
        std::cout << " (Converged!)" << std::endl;
    } else {
        std::cout << std::endl;
    }

    // Increment iteration counter
    currentIteration++;
}


// --- OpenGL Buffer Update Functions ---

// Update the vertex buffer for data points
void updatePointVBO() {
    if (NUM_POINTS == 0) return;

    // Allocate buffer for position and color data (x,y,z, r,g,b)
    pointVertexData.assign(NUM_POINTS * 6, 0.0f);

    // Fill buffer with current data
    for (size_t i = 0; i < dataPoints.size(); ++i) {
        const auto& point = dataPoints[i];

        // Position data (x, y, z)
        pointVertexData[i * 6 + 0] = point.pos.x;
        pointVertexData[i * 6 + 1] = point.pos.y;
        pointVertexData[i * 6 + 2] = point.pos.z;

        // Determine color based on cluster assignment
        int colorIdx = (point.clusterId == -1 ||
                       point.clusterId >= K_CLUSTERS ||
                       point.clusterId >= clusterColorsVec.size() - 1)
                       ? clusterColorsVec.size() - 1  // Use gray for unassigned points
                       : point.clusterId;             // Use cluster color

        // Color data (r, g, b)
        const auto& color = clusterColorsVec[colorIdx];
        pointVertexData[i * 6 + 3] = color.r;
        pointVertexData[i * 6 + 4] = color.g;
        pointVertexData[i * 6 + 5] = color.b;
    }

    // Upload data to GPU
    glBindBuffer(GL_ARRAY_BUFFER, dataPointsVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, pointVertexData.size() * sizeof(float), pointVertexData.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Update the vertex buffer for centroids
void updateCentroidVBO() {
    if (K_CLUSTERS == 0) return;

    // Allocate buffer for position and color data (x,y,z, r,g,b)
    centroidVertexData.assign(K_CLUSTERS * 6, 0.0f);

    // Fill buffer with current data
    for (int k = 0; k < K_CLUSTERS; ++k) {
        // Safety check
        if (k >= centroids.size() || k >= clusterColorsVec.size() - 1) continue;

        const auto& centroid = centroids[k];

        // Position data (x, y, z)
        centroidVertexData[k * 6 + 0] = centroid.pos.x;
        centroidVertexData[k * 6 + 1] = centroid.pos.y;
        centroidVertexData[k * 6 + 2] = centroid.pos.z;

        // Make centroids brighter than their cluster color for visibility
        glm::vec3 color = clusterColorsVec[k];
        color = glm::min(glm::vec3(1.0f), color + glm::vec3(0.3f));

        // Color data (r, g, b)
        centroidVertexData[k * 6 + 3] = color.r;
        centroidVertexData[k * 6 + 4] = color.g;
        centroidVertexData[k * 6 + 5] = color.b;
    }

    // Upload data to GPU
    // Note: Using glBufferData instead of glBufferSubData for flexibility
    // This allows handling cases where centroids might not exist initially
    glBindBuffer(GL_ARRAY_BUFFER, centroidsVBO);
    glBufferData(GL_ARRAY_BUFFER, centroidVertexData.size() * sizeof(float),
                centroidVertexData.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


// --- Camera Control Functions ---

// Update camera position based on orbital camera parameters
void updateCameraVectors() {
    // Calculate the direction vector from camera to target
    glm::vec3 front;
    front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    front.y = sin(glm::radians(cameraPitch));
    front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    // Note: We don't need to normalize front since we use it for direction only

    // Calculate camera position by moving back from target along front vector
    cameraPos.x = cameraTarget.x - cameraDistance * front.x;
    cameraPos.y = cameraTarget.y - cameraDistance * front.y;
    cameraPos.z = cameraTarget.z - cameraDistance * front.z;
}


// --- GLFW Input Callback Functions ---

// Handle window resize events
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Update OpenGL viewport to match new window dimensions
    glViewport(0, 0, width, height);

    // Update internal window size variables
    WINDOW_WIDTH = width;
    WINDOW_HEIGHT = height;
}

// Handle keyboard input
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        // Exit application
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        }

        // Perform one step of K-means algorithm
        if (key == GLFW_KEY_SPACE) {
            stepKMeans();
            updatePointVBO();    // Update colors based on new cluster assignments
            updateCentroidVBO(); // Update centroid positions
        }

        // Reset data and algorithm state
        if (key == GLFW_KEY_R) {
            std::cout << "--- Resetting ---" << std::endl;

            // Generate new random data and centroids
            generateRandomData3D();
            initializeCentroids3D();

            // Reset cluster assignments
            for (auto& point : dataPoints) {
                point.clusterId = -1;
            }

            // Reset algorithm state
            currentIteration = 0;
            converged = false;

            // Perform initial cluster assignment
            assignPointsToClusters3D();

            // Update visualization
            updatePointVBO();
            updateCentroidVBO();

            std::cout << "Data reset. Press SPACE to start." << std::endl;
        }
    }
}

// Handle mouse button events
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            leftMouseButtonPressed = true;
        } else if (action == GLFW_RELEASE) {
            leftMouseButtonPressed = false;
        }
    }
}

// Handle mouse movement
void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    // Handle first mouse input to avoid sudden jumps
    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
        return;
    }

    // Calculate mouse movement since last frame
    float xoffset = xpos - lastMouseX;
    float yoffset = lastMouseY - ypos; // Reversed: y-coordinates go from bottom to top

    // Update last position
    lastMouseX = xpos;
    lastMouseY = ypos;

    // Only rotate camera if left mouse button is pressed
    if (leftMouseButtonPressed) {
        // Apply sensitivity to make movement smoother
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        // Update camera angles
        cameraYaw += xoffset;   // Horizontal rotation
        cameraPitch += yoffset; // Vertical rotation

        // Constrain pitch to avoid flipping
        if (cameraPitch > 89.0f) cameraPitch = 89.0f;
        if (cameraPitch < -89.0f) cameraPitch = -89.0f;

        // Update camera position based on new angles
        updateCameraVectors();
    }
}

// Handle mouse scroll wheel
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Adjust camera distance (zoom)
    cameraDistance -= (float)yoffset * scrollSensitivity;

    // Constrain distance to reasonable limits
    if (cameraDistance < 1.0f) {
        cameraDistance = 1.0f; // Minimum zoom (closest)
    }
    if (cameraDistance > DATA_RANGE_MAX * 5.0f) {
        cameraDistance = DATA_RANGE_MAX * 5.0f; // Maximum zoom (farthest)
    }

    // Update camera position based on new distance
    updateCameraVectors();
}


// Main application entry point
int main() {
    // --- GLFW Initialization ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Request OpenGL 3.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // macOS compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        "KMeans 3D ModernGL",
        NULL, NULL
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Set current context and register callbacks
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // --- GLAD Initialization (OpenGL function loader) ---
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // --- OpenGL Global State Configuration ---
    glEnable(GL_DEPTH_TEST);           // Enable depth testing for 3D rendering
    glEnable(GL_PROGRAM_POINT_SIZE);   // Enable setting point size in shaders
    glEnable(GL_BLEND);                // Enable alpha blending for smooth points
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Standard alpha blending

    // --- Shader Compilation ---
    Shader pointShader("shaders/point.vert", "shaders/point.frag"); // For points and centroids
    Shader axisShader("shaders/axis.vert", "shaders/axis.frag");    // For coordinate axes

    // --- Coordinate Axes Setup ---
    // Create vertex data for X, Y, Z axes (from origin to max range)
    float axisVerts[] = {
        // X-axis (red)
        0.0f, 0.0f, 0.0f, DATA_RANGE_MAX, 0.0f, 0.0f,
        // Y-axis (green)
        0.0f, 0.0f, 0.0f, 0.0f, DATA_RANGE_MAX, 0.0f,
        // Z-axis (blue)
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, DATA_RANGE_MAX
    };

    // Create and configure VAO/VBO for axes
    glGenVertexArrays(1, &axisVAO);
    glGenBuffers(1, &axisVBO);
    glBindVertexArray(axisVAO);
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axisVerts), axisVerts, GL_STATIC_DRAW);

    // Configure vertex attributes (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Unbind to prevent accidental modification
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // --- Data Points Setup ---
    // Create and configure VAO/VBO for data points
    glGenVertexArrays(1, &dataPointsVAO);
    glGenBuffers(1, &dataPointsVBO);
    glBindVertexArray(dataPointsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, dataPointsVBO);

    // Allocate buffer (will be filled by updatePointVBO)
    glBufferData(GL_ARRAY_BUFFER, NUM_POINTS * 6 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Configure vertex attributes (position and color)
    // Position attribute (x, y, z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind to prevent accidental modification
    glBindVertexArray(0);

    // --- Centroids Setup ---
    // Create and configure VAO/VBO for centroids
    glGenVertexArrays(1, &centroidsVAO);
    glGenBuffers(1, &centroidsVBO);
    glBindVertexArray(centroidsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, centroidsVBO);

    // Allocate buffer (will be filled by updateCentroidVBO)
    glBufferData(GL_ARRAY_BUFFER, K_CLUSTERS * 6 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Configure vertex attributes (position and color)
    // Position attribute (x, y, z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind to prevent accidental modification
    glBindVertexArray(0);


    // --- Initialize K-means Algorithm Data ---
    generateRandomData3D();      // Generate random data points
    generateClusterColors();     // Create colors for each cluster
    initializeCentroids3D();     // Initialize cluster centroids
    assignPointsToClusters3D();  // Perform initial cluster assignment

    // --- Initialize Visualization ---
    updatePointVBO();            // Upload data points to GPU
    updateCentroidVBO();         // Upload centroids to GPU
    updateCameraVectors();       // Set initial camera position

    // --- Display Application Instructions ---
    std::cout << "KMeans 3D Visualizer (ModernGL)" << std::endl;
    std::cout << "Mouse Drag Left: Orbit camera" << std::endl;
    std::cout << "Mouse Scroll: Zoom camera" << std::endl;
    std::cout << "SPACE: Next KMeans iteration" << std::endl;
    std::cout << "R: Reset data" << std::endl;
    std::cout << "ESC: Exit" << std::endl;

    // --- Main Render Loop ---
    while (!glfwWindowShouldClose(window)) {
        // Input handling is done via callbacks

        // --- Clear the screen ---
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);  // Dark blue-gray background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- Setup View and Projection Matrices ---
        // Perspective projection matrix
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),                          // Field of view (45 degrees)
            (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT,   // Aspect ratio
            0.1f,                                         // Near clipping plane
            DATA_RANGE_MAX * 10.0f                        // Far clipping plane
        );

        // View matrix (camera position and orientation)
        glm::mat4 view = glm::lookAt(
            cameraPos,       // Camera position
            cameraTarget,    // Look-at target
            cameraUp         // Up vector
        );

        // Model matrix (identity for world space objects)
        glm::mat4 model = glm::mat4(1.0f);

        // --- Render Coordinate Axes ---
        axisShader.use();
        // Combined MVP matrix for axes
        glm::mat4 axisMVP = projection * view * glm::mat4(1.0f);
        axisShader.setMat4("MVP", axisMVP);
        glBindVertexArray(axisVAO);

        // X-axis (red)
        axisShader.setVec3("lineColor", 1.0f, 0.0f, 0.0f);
        glDrawArrays(GL_LINES, 0, 2);

        // Y-axis (green)
        axisShader.setVec3("lineColor", 0.0f, 1.0f, 0.0f);
        glDrawArrays(GL_LINES, 2, 2);

        // Z-axis (blue)
        axisShader.setVec3("lineColor", 0.0f, 0.0f, 1.0f);
        glDrawArrays(GL_LINES, 4, 2);

        glBindVertexArray(0);

        // --- Render Data Points ---
        if (NUM_POINTS > 0) {
            pointShader.use();

            // Set transformation matrices
            pointShader.setMat4("projection", projection);
            pointShader.setMat4("view", view);
            pointShader.setMat4("model", model);

            // Set point size
            pointShader.setFloat("u_pointSize", POINT_SIZE_DATA);

            // Draw all data points
            glBindVertexArray(dataPointsVAO);
            glDrawArrays(GL_POINTS, 0, NUM_POINTS);
            glBindVertexArray(0);
        }

        // --- Render Centroids ---
        if (K_CLUSTERS > 0 && !centroids.empty()) {
            pointShader.use();

            // Set transformation matrices (redundant if shader is already active, but good practice)
            pointShader.setMat4("projection", projection);
            pointShader.setMat4("view", view);
            pointShader.setMat4("model", model);

            // Set larger point size for centroids
            pointShader.setFloat("u_pointSize", POINT_SIZE_CENTROID);

            // Draw all centroids
            glBindVertexArray(centroidsVAO);
            glDrawArrays(GL_POINTS, 0, K_CLUSTERS > centroids.size() ? centroids.size() : K_CLUSTERS);
            glBindVertexArray(0);
        }

        // --- Swap buffers and poll events ---
        glfwSwapBuffers(window);  // Display the rendered frame
        glfwPollEvents();         // Process input events
    }

    // --- Cleanup Resources ---
    // Delete OpenGL objects to free GPU memory
    glDeleteVertexArrays(1, &dataPointsVAO);
    glDeleteBuffers(1, &dataPointsVBO);
    glDeleteVertexArrays(1, &centroidsVAO);
    glDeleteBuffers(1, &centroidsVBO);
    glDeleteVertexArrays(1, &axisVAO);
    glDeleteBuffers(1, &axisVBO);

    // Terminate GLFW and free resources
    glfwTerminate();

    return 0;
}