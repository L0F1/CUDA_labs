
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK 1024

__global__ void bitonicSortKernel(int* d_arr, int n, int k, int j) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x,
	    map = (tid / (1 << (j - 1)))*(1 << j) + (tid % (1 << (j - 1))),
		pos = (map / (1 << k)) % 2,
		e1 = (pos == 0) ? map : (map + (1 << (j - 1))),
		e2 = (pos == 0) ? (map + (1 << (j - 1))) : map;

	atomicMin(&d_arr[e1], atomicMax(&d_arr[e2], d_arr[e1]));
	__syncthreads();
}

void bitonicSortGpu(int* arr, int N) {
	int* d_arr;
	int logn2 = (int)(log(N) / log(2));
	cudaMalloc((void **)&d_arr, sizeof(int)*N);
	cudaMemcpy(d_arr, arr, sizeof(int) * N, cudaMemcpyHostToDevice);
	int blocks = ((N / 2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	for (int k = 1; k <= logn2; k++) {
		for (int j = k; j > 0; j--) {
			bitonicSortKernel<<<blocks, THREADS_PER_BLOCK >>>(d_arr, N, k, j);
		}
	}

	cudaDeviceSynchronize();
	cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_arr);
}

void bitonicSortCpu(int* arr, int N) {
	int logn2 = (int)(log(N) / log(2));

	for (int k = 1; k <= logn2; k++) {
		for (int j = k; j > 0; j--) {
			for (int i = 0; i < N/2; i++)
			{
				int map = (i / (1 << (j - 1)))*(1 << j) + (i % (1 << (j - 1))),
					pos = (map / (1 << k)) % 2,
					e1 = (pos == 0) ? map : (map + (1 << (j - 1))),
					e2 = (pos == 0) ? (map + (1 << (j - 1))) : map;

				if (arr[e1] > arr[e2]) {
					float temp = arr[e1];
					arr[e1] = arr[e2];
					arr[e2] = temp;
				}
			}
		}
	}
}

bool isSorted(int* arr, int N) {
	for (int i = 1; i < N; i++)
	{
		if (arr[i] < arr[i - 1])
			return false;
	}
	return true;
}

int* fillArray(int N) {
	int *arr = new int[N];
	for (int i = 0; i < N; i++)
	{
		arr[i] = N - i;
	}
	return arr;
}

void showArray(int* arr, int N) {
	for (int i = 0; i < N; i++)
	{
		std::cout << arr[i] << " ";
	}
}

int main() {

	int N;
	std::cout << "Enter size of the vector: ";
	std::cin >> N;

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int* arr = fillArray(N);
	bitonicSortGpu(arr, N);

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "=====================   GPU   =====================\n";
	std::cout << "DEVICE GPU compute time: " << double(gpuTime) / pow(10, 3) << " seconds\n";
	printf("Is array sorted?: %s", isSorted(arr, N) ? "YES\n\n" : "NO\n\n");

	using micro = std::chrono::microseconds;
	auto start2 = std::chrono::high_resolution_clock::now();

	delete[] arr;
	arr = fillArray(N);
	bitonicSortCpu(arr, N);

	//showArray(arr, N);

	auto stop2 = std::chrono::high_resolution_clock::now();
	auto cpuTime = std::chrono::duration_cast<micro>(stop2 - start2).count();

	std::cout << "=====================   CPU   =====================\n";
	std::cout << "HOST CPU compute time: " << double(cpuTime) / pow(10, 6) << " seconds\n";
	printf("Is array sorted?: %s", isSorted(arr, N) ? "YES\n\n" : "NO\n\n");

	return 0;
}