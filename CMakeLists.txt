cmake_minimum_required(VERSION 3.19)
project(sigilvg C)
if(APPLE)
    enable_language(OBJC)
endif()
include(FetchContent)

# --- WGVK (WebGPU on Vulkan) with simple_wgsl shader compiler ---
set(WGVK_WGSL_SUPPORT "SIMPLE_WGSL" CACHE STRING "" FORCE)
set(WGVK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WGVK_BUILD_GLSL_SUPPORT OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WGSL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

set(WGVK_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/../WGVK")
if(EXISTS "${WGVK_LOCAL}/CMakeLists.txt")
    add_subdirectory("${WGVK_LOCAL}" "${CMAKE_CURRENT_BINARY_DIR}/wgvk")
    set(WGVK_SOURCE_DIR "${WGVK_LOCAL}")
else()
    FetchContent_Declare(wgvk
        GIT_REPOSITORY https://github.com/manuel5975p/WGVK.git
        GIT_TAG        master
        GIT_SHALLOW    True
    )
    FetchContent_MakeAvailable(wgvk)
    set(WGVK_SOURCE_DIR "${wgvk_SOURCE_DIR}")
endif()

# Emscripten emcc treats -Wint-conversion as error; WGVK has a benign VkFence=NULL init.
if(EMSCRIPTEN AND TARGET simple_wgsl_c_api)
    target_compile_options(simple_wgsl_c_api PRIVATE -Wno-int-conversion -Wno-constant-conversion)
endif()

if(NOT EMSCRIPTEN)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_CLIENT_LIBRARY "none" CACHE STRING "" FORCE)
    set(GLFW_INCLUDE_NONE ON CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP True
    )
    FetchContent_MakeAvailable(glfw)
endif()

set(STB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stb")
if(NOT EXISTS "${STB_DIR}/stb_truetype.h")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h"
        "${STB_DIR}/stb_truetype.h"
    )
endif()
if(NOT EXISTS "${STB_DIR}/stb_image_write.h")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
        "${STB_DIR}/stb_image_write.h"
    )
endif()
if(NOT EXISTS "${STB_DIR}/stb_image.h")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${STB_DIR}/stb_image.h"
    )
endif()

add_library(sigilvg INTERFACE)
target_include_directories(sigilvg INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${STB_DIR}"
)
target_link_libraries(sigilvg INTERFACE wgvk)
target_compile_features(sigilvg INTERFACE c_std_11)

if(NOT EMSCRIPTEN)
    add_executable(sigilvg_test_parse tests/test_parse.c)
    target_link_libraries(sigilvg_test_parse PRIVATE sigilvg)
    target_include_directories(sigilvg_test_parse PRIVATE "${STB_DIR}")
    if(NOT MSVC)
        target_link_libraries(sigilvg_test_parse PRIVATE m)
    endif()

    add_executable(sigilvg_headless examples/headless.c)
    target_link_libraries(sigilvg_headless PRIVATE sigilvg)
    target_include_directories(sigilvg_headless PRIVATE "${STB_DIR}")
    if(NOT MSVC)
        target_link_libraries(sigilvg_headless PRIVATE m)
    endif()

    add_executable(sigilvg_viewer examples/viewer.c)
    if(APPLE)
        set_source_files_properties(examples/viewer.c PROPERTIES LANGUAGE OBJC)
    endif()
    target_link_libraries(sigilvg_viewer PRIVATE sigilvg glfw)
    target_include_directories(sigilvg_viewer PRIVATE
        "${STB_DIR}"
        "${WGVK_SOURCE_DIR}/examples"
    )
    if(NOT MSVC)
        target_link_libraries(sigilvg_viewer PRIVATE m)
    endif()

    configure_file(shaders/sigil_vertex.wgsl ${CMAKE_CURRENT_BINARY_DIR}/sigil_vertex.wgsl COPYONLY)
    configure_file(shaders/sigil_fragment.wgsl ${CMAKE_CURRENT_BINARY_DIR}/sigil_fragment.wgsl COPYONLY)
    configure_file(shaders/sigil_prepare_compute.wgsl ${CMAKE_CURRENT_BINARY_DIR}/sigil_prepare_compute.wgsl COPYONLY)
    configure_file(shaders/sigil_bake_gradient.wgsl ${CMAKE_CURRENT_BINARY_DIR}/sigil_bake_gradient.wgsl COPYONLY)

    # --- PlutoVG / PlutoSVG reference renderer ---
    set(PLUTOVG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PLUTOSVG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    if(NOT TARGET plutovg::plutovg)
        FetchContent_Declare(plutovg
            URL https://github.com/sammycage/plutovg/archive/refs/tags/v1.3.2.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP True
        )
        FetchContent_MakeAvailable(plutovg)
    endif()
    if(NOT TARGET plutosvg::plutosvg)
        FetchContent_Declare(plutosvg
            URL https://github.com/sammycage/plutosvg/archive/refs/tags/v0.0.7.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP True
        )
        FetchContent_Populate(plutosvg)
        file(REMOVE_RECURSE "${plutosvg_SOURCE_DIR}/plutovg")
        file(MAKE_DIRECTORY "${plutosvg_SOURCE_DIR}/plutovg")
        file(WRITE "${plutosvg_SOURCE_DIR}/plutovg/CMakeLists.txt" "# stub — plutovg already added\n")
        add_subdirectory("${plutosvg_SOURCE_DIR}" "${plutosvg_BINARY_DIR}" EXCLUDE_FROM_ALL)
    endif()
    set(HAVE_PLUTOSVG TRUE)

    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}" CACHE BOOL "" FORCE)

    if(HAVE_PLUTOSVG)
        add_executable(sigilvg_test_ref tests/test_reference.c)
        target_link_libraries(sigilvg_test_ref PRIVATE sigilvg plutosvg::plutosvg plutovg::plutovg)
        target_include_directories(sigilvg_test_ref PRIVATE "${STB_DIR}")
        if(NOT MSVC)
            target_link_libraries(sigilvg_test_ref PRIVATE m)
        endif()
    endif()

    add_executable(sigilvg_headless_batch examples/headless_batch.c)
    target_link_libraries(sigilvg_headless_batch PRIVATE sigilvg)
    target_include_directories(sigilvg_headless_batch PRIVATE "${STB_DIR}")
    if(NOT MSVC)
        target_link_libraries(sigilvg_headless_batch PRIVATE m)
    endif()

    add_executable(sigilvg_batch_pixel tests/batch_pixel_test.c)
    target_link_libraries(sigilvg_batch_pixel PRIVATE sigilvg)
    target_include_directories(sigilvg_batch_pixel PRIVATE "${STB_DIR}")
    if(NOT MSVC)
        target_link_libraries(sigilvg_batch_pixel PRIVATE m)
    endif()

endif()

# --- Emscripten web demo ---
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")

    add_executable(sigilvg_web_demo examples/web_demo.c)
    target_link_libraries(sigilvg_web_demo PRIVATE sigilvg)
    target_include_directories(sigilvg_web_demo PRIVATE
        "${STB_DIR}"
    )
    target_link_libraries(sigilvg_web_demo PRIVATE m)

    # emdawnwebgpu port + GLFW needed at link time
    target_link_options(sigilvg_web_demo PRIVATE
        "SHELL:--use-port=emdawnwebgpu"
        "SHELL:-sUSE_GLFW=3"
    )

    # Embed shader files into the WASM virtual filesystem
    target_link_options(sigilvg_web_demo PRIVATE
        "SHELL:--embed-file ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sigil_vertex.wgsl@sigil_vertex.wgsl"
        "SHELL:--embed-file ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sigil_fragment.wgsl@sigil_fragment.wgsl"
        "SHELL:--embed-file ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sigil_prepare_compute.wgsl@sigil_prepare_compute.wgsl"
        "SHELL:--embed-file ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sigil_bake_gradient.wgsl@sigil_bake_gradient.wgsl"
    )

    # Custom HTML shell with drag-and-drop UI
    target_link_options(sigilvg_web_demo PRIVATE
        "SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/examples/shell.html"
    )

    # Allow enough memory for SVG parsing + GPU texture staging
    target_link_options(sigilvg_web_demo PRIVATE
        "SHELL:-sALLOW_MEMORY_GROWTH=1"
        "SHELL:-sMAXIMUM_MEMORY=512MB"
        "SHELL:-sASYNCIFY=2"
    )

    # Export functions for JS interop (drag-and-drop SVG loading)
    target_link_options(sigilvg_web_demo PRIVATE
        "SHELL:-sEXPORTED_FUNCTIONS=['_main','_web_demo_load_svg']"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=['ccall']"
    )
endif()
