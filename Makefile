SOURCE := main.cu
BIN := prog
FLAGS := -arch=sm_75 -Xcompiler -fopenmp
COMPILER := nvcc

all:
	${COMPILER} ${FLAGS} ${SOURCE} -o ${BIN}

clean:
	rm -rf prog