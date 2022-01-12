
#include "cuda_runtime.h"
#include <cuda.h> 
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>


__global__
void cos_gpu(float *arg, float *res) {

	int b = blockIdx.x * blockDim.x;
	res[threadIdx.x + b] = cosf(arg[threadIdx.x + b]);
}

float randFloat(float min, float max) {
	srand((unsigned int)time(NULL));
	return  (max - min) * ((((float)rand()) / (float)RAND_MAX)) + min;
}

void gpu_compute(int N) {

	float *host_args = new float[N];
	float *host_res = new float[N];

	for (auto i = 0; i < N; i++)
		host_args[i] = randFloat(0, 1);

	float *device_args, *device_res;
	const int size = N * sizeof(float);

	// начало отсчета времени
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// выделение памяти под переменные в device
	cudaMalloc(&device_args, size);
	cudaMalloc(&device_res, size);

	// копироваание переменной из host в device
	cudaMemcpy(device_args, host_args, size, cudaMemcpyHostToDevice);


	// вычисление функции
	cos_gpu <<<1, N >>> (device_args, device_res);

	// дожидаемся выполнения
	cudaThreadSynchronize();

	// копироваание переменной из device в host
	cudaMemcpy(host_res, device_res, size, cudaMemcpyDeviceToHost);

	// конец отсчета времени
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	// очистка памяти
	cudaFree(device_args);
	cudaFree(device_res);
	delete[] host_args;
	delete[] host_res;

	std::cout << "\n\n=====================   GPU   =====================\n";
	std::cout << "DEVICE GPU compute time: " << gpuTime * 1000 << " microseconds\n\n";
}

void cpu_compute(int N) {

	float *host_args = new float[N];
	float *host_res = new float[N];

	for (auto i = 0; i < N; i++)
		host_args[i] = randFloat(0, 1);

	using micro = std::chrono::microseconds;
	auto start = std::chrono::high_resolution_clock::now();

	// вычисление функции
	for (auto i = 0; i < N; i++)
		host_res[i] = cosf(host_args[i]);

	auto stop = std::chrono::high_resolution_clock::now();
	auto cpuTime = std::chrono::duration_cast<micro>(stop - start).count();

	delete[] host_args;
	delete[] host_res;

	std::cout << "\n\n=====================   CPU   =====================\n";
	std::cout << "HOST CPU compute time: " << cpuTime << " microseconds\n\n";
}

int main()
{
	int count;

	std::cout << "Enter the number of cosine calculations: ";
	std::cin >> count;

	gpu_compute(count);
	cpu_compute(count);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}