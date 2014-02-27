CC              := gcc
CFLAGS          := -ggdb `pkg-config --cflags opencv` -Wall -Wextra -pedantic -std=c99
OBJECTS         := 
LIBRARIES       := `pkg-config --libs opencv` -lm -pthread
SRC             := motion-sensor-opencv.c
BIN             := $(subst .c,,$(SRC))

.PHONY: all clean

all: test

test: 
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LIBRARIES)
        
clean:
	rm -f $(BIN)
