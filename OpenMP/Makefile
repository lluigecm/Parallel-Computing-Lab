CC = gcc
CFLAGS = -O2 -Wall

all: gaussian

gaussian: main.c gaussian.c
	$(CC) $(CFLAGS) -o gaussian main.c gaussian.c

clean:
	rm -f gaussian