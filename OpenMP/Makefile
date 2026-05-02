CC     = gcc
CFLAGS = -O2 -Wall -fopenmp

all: serial openmp tiling

serial: serial.c gaussian.c
	$(CC) $(CFLAGS) -o serial serial.c gaussian.c -lm

openmp: openmp.c gaussian.c
	$(CC) $(CFLAGS) -o openmp openmp.c gaussian.c -lm

tiling: tiling.c gaussian.c
	$(CC) $(CFLAGS) -o tiling tiling.c gaussian.c -lm

clean:
	rm -f serial openmp tiling output_*.png