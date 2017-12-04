# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3
ARCH   	:= -march=native

# Language standard
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS :=
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
.DEFAULT_GOAL := all

# executables
EXEC := generate_data

.PHONY: debug
debug : OPT  := -O0 -g -G
debug : LDFLAGS :=
debug : ARCH :=
debug : $(EXEC)

all : $(EXEC)

generate_data: main.cu heatEquation.hpp
	module load cuda;nvcc -o generate_data $(OPT) main.cu -ccbin $(BIN)
