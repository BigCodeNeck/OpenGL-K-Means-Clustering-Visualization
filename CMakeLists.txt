# OpenGL K-Means Clustering Visualization - CMake Configuration
# This file configures the build process for the OpenGL K-Means application

# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and basic info
project(OpenGLCube)

# --- C++ Standard Configuration ---
# Using C++17 for modern features like structured bindings and std::optional
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)  # Require exact C++17 standard
set(CMAKE_CXX_EXTENSIONS False)        # Disable compiler-specific extensions

# --- Output Directory Configuration ---
# Place the executable in the build directory for easy access
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- Required External Dependencies ---
# Find the OpenGL package which is required for rendering
find_package(OpenGL REQUIRED)

# --- Source Files ---
# List all source files needed for the application
set(SOURCES
    src/main.cpp                # Main application code
    external/glad/src/glad.c    # OpenGL function loader implementation
)

# --- Create Executable Target ---
add_executable(${PROJECT_NAME} ${SOURCES})

# --- Include Directories ---
# Specify all the include paths needed for header files
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}                     # Project root directory
    ${CMAKE_SOURCE_DIR}/external/glad/include  # GLAD OpenGL loader headers
    ${CMAKE_SOURCE_DIR}/external/glfw/include  # GLFW window management headers
    ${CMAKE_SOURCE_DIR}/external/glm           # GLM math library headers
    ${CMAKE_SOURCE_DIR}/external               # Other external dependencies
)

# --- Link Libraries ---
# Link against OpenGL libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENGL_LIBRARIES}
)

# --- Platform-Specific Configuration ---
# Different platforms require different linking configurations
if(WIN32)
    # Windows-specific configuration
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/external/glfw/lib-vc2017/glfw3.lib  # Pre-built GLFW library
        opengl32.lib                                           # Windows OpenGL library
    )
elseif(APPLE)
    # macOS-specific configuration
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Cocoa"       # macOS window management
        "-framework IOKit"       # macOS I/O kit for input handling
        "-framework CoreVideo"   # macOS video handling
        "-framework OpenGL"      # macOS OpenGL framework
    )
else()
    # Linux-specific configuration
    target_link_libraries(${PROJECT_NAME} PRIVATE
        glfw    # GLFW library (assumed to be installed system-wide on Linux)
        GL      # OpenGL library
    )
endif()

# --- Shader Files Handling ---
# Copy shader files to the build directory so they can be loaded at runtime
if(EXISTS "${CMAKE_SOURCE_DIR}/shaders")
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/Debug/shaders
        COMMENT "Copying shaders directory to build output"
        VERBATIM
    )
endif()

# --- Build Configuration Summary ---
# Print information about the build configuration for debugging
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "GLAD source: ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c")
message(STATUS "GLFW library: ${CMAKE_SOURCE_DIR}/external/glfw/lib-vc2017/glfw3.lib")
message(STATUS "GLM include: ${CMAKE_SOURCE_DIR}/external/glm")