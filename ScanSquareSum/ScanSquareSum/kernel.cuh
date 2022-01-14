#include "cuda_runtime.h"

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo, bool isSumScan);
__global__ void prescan_large(int *output, int *input, int n, int *sums, bool isSumScan);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);