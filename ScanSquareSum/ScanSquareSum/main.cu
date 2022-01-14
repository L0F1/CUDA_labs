#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;


int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool isSumScan);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool isSumScan);

int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

void cpuScan(int* output, int* input, int length) {
	output[0] = pow(input[0], 2);
	for (int j = 1; j < length; ++j)
	{
		output[j] = pow(input[j - 1], 2) + output[j - 1];
	}
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool isSumScan) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums, isSumScan);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, true);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, true);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool isSumScan) {
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo, isSumScan);
}

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool isSumScan) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, isSumScan);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, isSumScan);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, isSumScan);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void gpuScan(int *output, int *input, int length) {
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);


	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, false);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, false);
	}


	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}


int main() {

	int N;

	std::cout << "\nEnter size of the vector: ";
	std::cin >> N;

	// generate values
	int *in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = 2;
	}

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int *out = new int[N];
	gpuScan(out, in, N);

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, start, stop);

	long lastSquare = N * N;

	std::cout << "\nSum of squares is - " << out[N - 1] + pow(in[N -1], 2) << "\n";
	std::cout << "=====================   GPU   =====================\n";
	std::cout << "DEVICE GPU compute time: " << double(gpuTime) / pow(10, 3) << " seconds\n\n";

	using micro = std::chrono::microseconds;
	auto start2 = std::chrono::high_resolution_clock::now();

	delete[] out;
	out = new int[N];
	cpuScan(out, in, N);

	auto stop2 = std::chrono::high_resolution_clock::now();
	auto cpuTime = std::chrono::duration_cast<micro>(stop2 - start2).count();

	std::cout << "Sum of squares is - " << out[N - 1] << "\n";
	std::cout << "=====================   CPU   =====================\n";
	std::cout << "HOST CPU compute time: " << double(cpuTime) / pow(10, 6) << " seconds\n\n";

	return 0;
}