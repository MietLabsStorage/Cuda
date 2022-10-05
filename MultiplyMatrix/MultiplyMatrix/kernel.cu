#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>

constexpr auto threads_for_block = 1024;
constexpr auto matrix_len = 1024;

__global__ void multiply_with_matrix_config(const int* matrix_a, const int* matrix_b, int* matrix_c, const int size)
{
	const int block_x = static_cast<int>(blockIdx.x);
	const int block_y = static_cast<int>(blockIdx.y);
	const int thread_x = static_cast<int>(threadIdx.x);
	const int thread_y= static_cast<int>(threadIdx.y);

	int sum = 0;
	const int index_a = threads_for_block * block_x + thread_x;
	const int index_b = threads_for_block * block_y + thread_y;

	for (int k = 0; k < size; k++)
	{
		sum += matrix_a[index_a * size + k] * matrix_b[index_b + k * size];
	}

	const int index_c = size * index_a + index_b;
	matrix_c[index_c] = sum;
}

__global__ void multiply_with_line_config(const int* matrix_a, const int* matrix_b, int* matrix_c, const int size)
{
	const int block = static_cast<int>(blockIdx.x);
	const int thread = static_cast<int>(threadIdx.x);

	int sum = 0;
	const int index = threads_for_block * block + thread;

	const int index_a = index / matrix_len;
	const int index_b = index % matrix_len;

	for (int k = 0; k < size; k++)
	{
		sum += matrix_a[index_a * size + k] * matrix_b[index_b + k * size];
	}

	const int index_c = index;
	matrix_c[index_c] = sum;
}

bool check_matrix(const int* actual, const int* expected, const int n)
{
	for (auto i = 0; i < n; i++)
	{
		for (auto j = 0; j < n; j++)
		{
			if (actual[i * n + j] != expected[i * n + j])
			{
				return false;
			}
		} 
	}
	return true;
}

void multiply_matrix(const int* matrix_a, const int* matrix_b, const int* expected_c, const int size)
{
	const auto matrix_size = static_cast<unsigned long long>(size) * size * sizeof(int);
	const auto actual_c = static_cast<int*>(malloc(matrix_size));

	int* gpu_a = nullptr;
	int* gpu_b = nullptr;
	int* gpu_c = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&gpu_a), matrix_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_b), matrix_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_c), matrix_size);

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threads(threads_for_block, threads_for_block);
	dim3 blocks(matrix_len / threads.x, matrix_len / threads.y);

	cudaMemcpy(gpu_a, matrix_a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, matrix_b, matrix_size, cudaMemcpyHostToDevice);

	cudaEventRecord(start, nullptr);
	//multiply_with_matrix_config << <blocks, threads >> >(gpu_a, gpu_b, gpu_c, matrix_len);
	multiply_with_line_config << <matrix_len * matrix_len / threads_for_block, threads_for_block >> > (gpu_a, gpu_b, gpu_c, matrix_len);
	cudaEventRecord(stop, nullptr);
	cudaMemcpy(actual_c, gpu_c, matrix_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);

	printf("result of checking: %s\n", check_matrix(actual_c, expected_c, size) ? "true" : "false");
	//printf("time spent executing by the GPU (%d threads, %d blocks): %.5f ms\n", threads.x * threads.y, blocks.x * blocks.y, static_cast<double>(gpu_time));
	printf("time spent executing by the GPU (%d threads, %d blocks): %.5f ms\n", threads_for_block, matrix_len * matrix_len / threads_for_block, static_cast<double>(gpu_time));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
}

int main(int argc, char* argv[])
{
	constexpr auto matrix_size = static_cast<unsigned long long>(matrix_len) * matrix_len * sizeof(int);
	const auto a = static_cast<int*>(malloc(matrix_size));
	const auto b = static_cast<int*>(malloc(matrix_size));
	const auto c = static_cast<int*>(malloc(matrix_size));
	for (int i = 0; i < matrix_len; i++)
		for (int j = 0; j < matrix_len; j++)
		{
			const int k = matrix_len * i + j;
			a[k] = 1;
			b[k] = 1;
			c[k] = matrix_len;
		}

	multiply_matrix(a, b, c, matrix_len);

	free(a);
	free(b);
	free(c);

	return EXIT_SUCCESS;
}
