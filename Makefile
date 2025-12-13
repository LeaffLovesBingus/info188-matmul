SOURCE := main.cu
BIN := prog
FLAGS := -Xcompiler -fopenmp
COMPILER := nvcc

all:
	${COMPILER} ${FLAGS} ${SOURCE} -o ${BIN}

clean:
	rm -rf prog