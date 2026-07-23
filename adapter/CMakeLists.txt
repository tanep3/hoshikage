cmake_minimum_required(VERSION 3.16)

project(hoshikage_llama_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(hoshikage-llama-adapter SHARED
    hoshikage_llama_adapter.cpp
)

target_include_directories(hoshikage-llama-adapter
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama_cpp_local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama_cpp_local/common
)

target_link_directories(hoshikage-llama-adapter
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama_cpp_local/lib
)

target_link_libraries(hoshikage-llama-adapter
    PRIVATE
        llama
        llama-common
)

if(UNIX AND NOT APPLE)
    set_target_properties(hoshikage-llama-adapter PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
