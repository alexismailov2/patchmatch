cmake_minimum_required(VERSION 2.8)
project(patchmatch)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-O6 -Wall -ffast-math")
#set(CMAKE_CXX_FLAGS "-O6 -Wall -fast-math -msse2 -msse")

set(patchmatch_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE patchmatch_PUBLIC_INCLUDES ${patchmatch_PUBLIC_INCLUDE_DIR}/patchmatch/*.hpp)

include_directories(
   include)

file(GLOB patchmatch_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

add_library(${PROJECT_NAME}
   STATIC
   ${patchmatch_PUBLIC_INCLUDES}
   ${patchmatch_SOURCES})

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS})

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
add_executable(${PROJECT_NAME}_test
   test/test.cpp
)

target_link_libraries(${PROJECT_NAME}_test
   ${OpenCV_LIBS}
   ${PROJECT_NAME})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${patchmatch_PUBLIC_INCLUDES}")

install(TARGETS ${PROJECT_NAME}
   ARCHIVE DESTINATION ${LIBRARY_INSTALL_DESTINATION}/lib/${ANDROID_ABI}
   LIBRARY DESTINATION ${LIBRARY_INSTALL_DESTINATION}/lib/${ANDROID_ABI}
   PUBLIC_HEADER DESTINATION ${LIBRARY_INSTALL_DESTINATION}/include/${PROJECT_NAME})

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
   install(TARGETS ${PROJECT_NAME}_test
      RUNTIME DESTINATION ${LIBRARY_INSTALL_DESTINATION}/bin/${ANDROID_ABI})
endif()