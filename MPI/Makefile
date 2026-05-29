CC     = mpicc
CFLAGS = -O2 -Wall

all: mpi

mpi: mpi.c gaussian.c gaussian.h
	$(CC) $(CFLAGS) -o mpi mpi.c gaussian.c -lm

clean:
	rm -f mpi output_mpi_*.png
