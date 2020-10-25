cmake_minimum_required(VERSION 2.8)
project(patchmatch)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ffast-math")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ffast-math -msse2 -msse")
endif()

set(patchmatch_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE patchmatch_PUBLIC_INCLUDES ${patchmatch_PUBLIC_INCLUDE_DIR}/patchmatch/*.hpp)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
   add_definitions(
           -DNEON=1
           -DSSE=0)
else()
   find_package(OpenCV REQUIRED)
   add_definitions(
           -DNEON=0
           -DSSE=1)
endif()

include_directories(
   ${LIBRARY_INSTALL_DESTINATION}/include
   include)

file(GLOB_RECURSE patchmatch_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

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

add_executable(labeling_test
   test/labeling.cpp)

set(STD_FILESYSTEM)
if (NOT APPLE)
   set(STD_FILESYSTEM stdc++fs)
endif()

target_link_libraries(labeling_test
   ${STD_FILESYSTEM}
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