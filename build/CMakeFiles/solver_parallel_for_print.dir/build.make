# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/krzysztof/hpc/projekt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/krzysztof/hpc/projekt/build

# Include any dependencies generated for this target.
include CMakeFiles/solver_parallel_for_print.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/solver_parallel_for_print.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/solver_parallel_for_print.dir/flags.make

CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o: CMakeFiles/solver_parallel_for_print.dir/flags.make
CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o: ../src/solver_parallel_for_print.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/krzysztof/hpc/projekt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o -c /home/krzysztof/hpc/projekt/src/solver_parallel_for_print.cpp

CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/krzysztof/hpc/projekt/src/solver_parallel_for_print.cpp > CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.i

CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/krzysztof/hpc/projekt/src/solver_parallel_for_print.cpp -o CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.s

# Object files for target solver_parallel_for_print
solver_parallel_for_print_OBJECTS = \
"CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o"

# External object files for target solver_parallel_for_print
solver_parallel_for_print_EXTERNAL_OBJECTS =

solver_parallel_for_print: CMakeFiles/solver_parallel_for_print.dir/src/solver_parallel_for_print.cpp.o
solver_parallel_for_print: CMakeFiles/solver_parallel_for_print.dir/build.make
solver_parallel_for_print: /usr/lib/x86_64-linux-gnu/libtbb.so.2
solver_parallel_for_print: /home/krzysztof/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/benchmark-1.6.0-oviunlk6fy6d5i62o3g7tzggavvrw5ze/lib/libbenchmark.a
solver_parallel_for_print: /usr/lib/x86_64-linux-gnu/librt.so
solver_parallel_for_print: CMakeFiles/solver_parallel_for_print.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/krzysztof/hpc/projekt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable solver_parallel_for_print"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solver_parallel_for_print.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/solver_parallel_for_print.dir/build: solver_parallel_for_print

.PHONY : CMakeFiles/solver_parallel_for_print.dir/build

CMakeFiles/solver_parallel_for_print.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/solver_parallel_for_print.dir/cmake_clean.cmake
.PHONY : CMakeFiles/solver_parallel_for_print.dir/clean

CMakeFiles/solver_parallel_for_print.dir/depend:
	cd /home/krzysztof/hpc/projekt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/krzysztof/hpc/projekt /home/krzysztof/hpc/projekt /home/krzysztof/hpc/projekt/build /home/krzysztof/hpc/projekt/build /home/krzysztof/hpc/projekt/build/CMakeFiles/solver_parallel_for_print.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/solver_parallel_for_print.dir/depend

