cmake_minimum_required(VERSION 3.0)
project( OpticFlow )
find_package( OpenCV REQUIRED )
add_executable( OpticFlow main.cpp )
target_link_libraries( OpticFlow ${OpenCV_LIBS} )
