#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "dev_array.h"
#include <chrono>

using namespace std;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N) {
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(float *A, float *B, float *C, int N) {

	// Max threads per block is 1024
	int SIZE = N * N;
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (SIZE > 1024) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel <<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}

vector<float> gpu_compute(vector<float> h_A, vector<float> h_B, int N) {

	using micro = std::chrono::microseconds;
	auto start = std::chrono::high_resolution_clock::now();

	int SIZE = N * N;

	// Allocate memory on the host
	vector<float> h_C(SIZE);

	// Allocate memory on the device
	dev_array<float> d_A(SIZE);
	dev_array<float> d_B(SIZE);
	dev_array<float> d_C(SIZE);

	d_A.set(&h_A[0], SIZE);
	d_B.set(&h_B[0], SIZE);

	matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
	cudaDeviceSynchronize();

	d_C.get(&h_C[0], SIZE);
	cudaDeviceSynchronize();

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			std::cout << h_C[i * N + j] << " ";
		}
		std::cout << std::endl;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto gpuTime = std::chrono::duration_cast<micro>(stop - start).count();

	std::cout << "\n\n=====================   GPU   =====================\n";
	std::cout << "DEVICE GPU compute time: " << double(gpuTime) / pow(10, 6) << " seconds\n\n";

	return h_C;
}

float* cpu_compute(vector<float> h_A, vector<float> h_B, int N) {

	using micro = std::chrono::microseconds;
	auto start = std::chrono::high_resolution_clock::now();

	int SIZE = N * N;
	float *cpu_C;
	cpu_C = new float[SIZE];

	float sum;
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			sum = 0.f;
			for (int n = 0; n < N; n++) {
				sum += h_A[row * N + n] * h_B[n * N + col];
			}
			cpu_C[row * N + col] = sum;
		}
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto cpuTime = std::chrono::duration_cast<micro>(stop - start).count();

	std::cout << "\n\n=====================   CPU   =====================\n";
	std::cout << "HOST CPU compute time: " << cpuTime / pow(10, 6) << " seconds\n\n";

	return cpu_C;
}

int main()
{
	int N;

	std::cout << "Enter size of the square matrix: ";
	std::cin >> N;

	int SIZE = N*N;

	// Allocate memory on host
	vector<float> h_A(SIZE);
	vector<float> h_B(SIZE);

	// Initialize matrices on host
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = sin(i);
			h_B[i*N + j] = cos(j);
		}
	}

	vector<float> h_C = gpu_compute(h_A, h_B, N);
	float* cpu_C = cpu_compute(h_A, h_B, N);

	// Check result
	double err = 0; 
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++) {
			err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
		}
	}

	cout << "Error: " << err << endl;

	return 0;
}