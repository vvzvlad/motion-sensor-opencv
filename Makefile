CC              := gcc
CFLAGS          := -ggdb `pkg-config --cflags opencv` -Wall -Wextra -pedantic -std=c99
OBJECTS         := 
LIBRARIES       := `pkg-config --libs opencv` -lm

.PHONY: all clean

all: test

test: 
	$(CC) $(CFLAGS) -o `basename motion-sensor-opencv.c .c` motion-sensor-opencv.c $(LIBRARIES)
        
clean:
	rm -f *.o
