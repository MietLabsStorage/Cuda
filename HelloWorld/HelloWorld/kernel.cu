#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#define N (1024*1024)

__global__ void kernel(float* data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 2 * 3.1415926 * static_cast<float>(idx) / static_cast<float>(N);
	data[idx] = sinf(sqrtf(x));
}

void hello_world()
{
	auto a = static_cast<float*>(malloc(N * sizeof(float)));
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
	int deviceCount;
	cudaDeviceProp devProp{};
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d devices\n", deviceCount);
	for (int device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&devProp, device);
		printf("Device %d\n", device);
		printf("Compute capability : %d.%d\n", devProp.major, devProp.minor);
		printf("Name : %s\n", devProp.name);
		printf("Total Global Memory : %llu\n", devProp.totalGlobalMem);
		printf("Shared memory per block: %llu\n", devProp.sharedMemPerBlock);
		printf("Registers per block : %d\n", devProp.regsPerBlock);
		printf("Warp size : %d\n", devProp.warpSize);
		printf("Max threads per block : %d\n", devProp.maxThreadsPerBlock);
		printf("Total constant memory : %llu\n", devProp.totalConstMem);
	}
}

void time_tracker()
{
	//описываем переменные типа cudaEvent_t
	cudaEvent_t start, stop;

	float gpuTime = 0.0f;

	// создаем события начала и окончания выполнения ядра
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* dev = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&dev), N * sizeof(float));

	//привязываем событие start к данному месту
	cudaEventRecord(start, 0);
	// вызвать ядро
	kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
	//привязываем событие stop к данному месту
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaFree(dev);

	// запрашиваем время между событиями
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("time spent executing by the GPU: %.5f ms\n", gpuTime);
	// уничтожаем созданные события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{
	about_devices();

	system("pause");

	time_tracker();

	system("pause");

	hello_world();

	return 0;
}