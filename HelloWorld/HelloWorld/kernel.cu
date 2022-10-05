#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#define N (1024*1024)

__global__ void kernel(float* data)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const float x = 2 * static_cast<float>(3.1415926) * static_cast<float>(idx) / static_cast<float>(N);
	data[idx] = sinf(sqrtf(x));
}

void hello_world()
{
	const auto a = static_cast<float*>(malloc(N * sizeof(float)));
	float* dev = nullptr;
	// выделить память на GPU
	cudaMalloc(reinterpret_cast<void**>(&dev), N * sizeof(float));
	// конфигурация запуска N нитей
	kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
	// скопировать результаты в память CPU
	cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
	// освободить выделенную память
	cudaFree(dev);

	for (int idx = 0; idx < N; idx++)
		printf("a[%d] = %.5f\n", idx, a[idx]);

	free(a);
}

void about_devices()
{
	int device_count;
	cudaDeviceProp dev_prop{};
	cudaGetDeviceCount(&device_count);
	printf("Found %d devices\n", device_count);
	for (int device = 0; device < device_count; device++)
	{
		cudaGetDeviceProperties(&dev_prop, device);
		printf("Device %d\n", device);
		printf("Compute capability : %d.%d\n", dev_prop.major, dev_prop.minor);
		printf("Name : %s\n", dev_prop.name);
		printf("Total Global Memory : %llu\n", dev_prop.totalGlobalMem);
		printf("Shared memory per block: %llu\n", dev_prop.sharedMemPerBlock);
		printf("Registers per block : %d\n", dev_prop.regsPerBlock);
		printf("Warp size : %d\n", dev_prop.warpSize);
		printf("Max threads per block : %d\n", dev_prop.maxThreadsPerBlock);
		printf("Max threads dim : %d*%d*%d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
		printf("Total constant memory : %llu\n", dev_prop.totalConstMem);
		printf("SM count: %d\n", dev_prop.multiProcessorCount);
	}
}

void time_tracker(bool default_blocks_count)
{
	//описываем переменные типа cudaEvent_t
	cudaEvent_t start, stop;

	float gpu_time = 0.0f;

	// создаем события начала и окончания выполнения ядра
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* dev = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&dev), N * sizeof(float));

	//привязываем событие start к данному месту
	cudaEventRecord(start, nullptr);
	// вызвать ядро
	if (default_blocks_count)
	{
		kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
	}
	else
	{
		kernel << <dim3(16, 1), dim3(512, 1) >> > (dev);
	}
	//привязываем событие stop к данному месту
	cudaEventRecord(stop, nullptr);
	cudaEventSynchronize(stop);

	cudaFree(dev);

	// запрашиваем время между событиями
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("time spent executing by the GPU (%d blocks): %.5f ms\n", default_blocks_count ? (N / 512) : 16, static_cast<double>(gpu_time));
	// уничтожаем созданные события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{
	about_devices();

	system("pause");

	time_tracker(true);
	time_tracker(false);

	system("pause");

	hello_world();

	return 0;
}