#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

__global__ void multiply_vectors(const int* vector_a, const int* vector_b, int* vector_c, const int size, const int step)
{
	const int block_x = static_cast<int>(blockIdx.x);
	const int thread_x = static_cast<int>(threadIdx.x);

	const int index = (static_cast<int>(blockDim.x) * block_x + thread_x + step) % size;

	vector_c[index] = vector_a[index] * vector_b[index];
}

bool check_vector(const int* actual, const int* expected, const int n)
{
	for (auto i = 0; i < n; i++)
	{
		if (actual[i] != expected[i])
		{
			return false;
		}
	}
	return true;
}

void multiply_vector(const int* vector_a, const int* vector_b, const int* expected_c, const int size)
{
	const auto vector_size = static_cast<int>(size) * sizeof(int);
	const auto actual_c = static_cast<int*>(malloc(vector_size));

	int* gpu_a = nullptr;
	int* gpu_b = nullptr;
	int* gpu_c1 = nullptr;
	int* gpu_c2 = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&gpu_a), vector_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_b), vector_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_c1), vector_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_c2), vector_size);

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threads(blockDim.x);
	dim3 blocks(size / threads.x);

	cudaMemcpy(gpu_a, vector_a, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, vector_b, vector_size, cudaMemcpyHostToDevice);

	constexpr auto step1 = 0;
	cudaEventRecord(start, nullptr);
	multiply_vectors << <blocks, threads >> > (gpu_a, gpu_b, gpu_c1, size, step1);
	cudaEventRecord(stop, nullptr);
	cudaMemcpy(actual_c, gpu_c1, vector_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("result of checking: %s\n", check_vector(actual_c, expected_c, size) ? "true" : "false");
	printf("time spent executing by the GPU (%d threads, %d blocks, step %d): %.5f ms\n", threads.x, blocks.x, step1, static_cast<double>(gpu_time));

	constexpr auto step2 = 13;
	cudaEventRecord(start, nullptr);
	multiply_vectors << <blocks, threads >> > (gpu_a, gpu_b, gpu_c2, size, step2);
	cudaEventRecord(stop, nullptr);
	cudaMemcpy(actual_c, gpu_c2, vector_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("result of checking: %s\n", check_vector(actual_c, expected_c, size) ? "true" : "false");
	printf("time spent executing by the GPU (%d threads, %d blocks, step %d): %.5f ms\n", threads.x, blocks.x, step2, static_cast<double>(gpu_time));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c1);
	cudaFree(gpu_c2);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void T_matrix_global(const int* matrix_a, int* matrix_c, const int size)
{
	const int block_x = static_cast<int>(blockIdx.x);
	const int block_y = static_cast<int>(blockIdx.y);
	const int thread_x = static_cast<int>(threadIdx.x);
	const int thread_y = static_cast<int>(threadIdx.y);
	const int grid_x = static_cast<int>(gridDim.x);
	const int grid_y = static_cast<int>(gridDim.y);
	const int thbl_x = static_cast<int>(blockDim.x);
	const int thbl_y = static_cast<int>(blockDim.y);

	const int index_x = thbl_x * block_x % grid_x + thread_x % thbl_x;
	const int index_y = thbl_y * block_y % grid_y + thread_y % thbl_y;

	matrix_c[index_x * size + index_y] = matrix_a[index_y * size + index_x];
}

__global__ void T_matrix_global_shared(const int* matrix_a, int* matrix_c, const int size)
{
	const int block_x = static_cast<int>(blockIdx.x);
	const int block_y = static_cast<int>(blockIdx.y);
	const int thread_x = static_cast<int>(threadIdx.x);
	const int thread_y = static_cast<int>(threadIdx.y);
	const int grid_x = static_cast<int>(gridDim.x);
	const int grid_y = static_cast<int>(gridDim.y);
	const int thbl_x = static_cast<int>(blockDim.x);
	const int thbl_y = static_cast<int>(blockDim.y);

	

	const int index_x = thbl_x * block_x % grid_x + thread_x % thbl_x;
	const int index_y = thbl_y * block_y % grid_y + thread_y % thbl_y;

	const int index1 = index_x * size + index_y;
	const int index2 = index_y * size + index_x;

	__shared__ int s_A[size * size];
	s_A[index2] = matrix_a[index2];

	matrix_c[index1] = s_A[index2];
}

bool check_matrix(const int* actual, const int* expected, const int n)
{
	for (auto i = 0; i < n; i++)
	{
		for (auto j = 0; j < n; j++)
		{
			//std::cout << i << "-" << j << ":  " << actual[i * n + j] << " " << expected[i * n + j] << "\n";
			if (actual[i * n + j] != expected[i * n + j])
			{
				return false;
			}
		}
	}
	return true;
}

void T_matrix(const int* matrix_a, const int* expected_c, const int size)
{
	const auto matrix_size = static_cast<unsigned long long>(size) * size * sizeof(int);
	const auto actual_c = static_cast<int*>(malloc(matrix_size));

	int* gpu_a = nullptr;
	int* gpu_c = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&gpu_a), matrix_size);
	cudaMalloc(reinterpret_cast<void**>(&gpu_c), matrix_size);

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threads(blockDim.x, blockDim.y);
	dim3 blocks(size / threads.x, size / threads.y);

	cudaMemcpy(gpu_a, matrix_a, matrix_size, cudaMemcpyHostToDevice);

	cudaEventRecord(start, nullptr);
	T_matrix_global << <blocks, threads >> >(gpu_a, gpu_a, size);
	cudaEventRecord(stop, nullptr);
	cudaMemcpy(actual_c, gpu_a, matrix_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);

	printf("result of checking: %s\n", check_matrix(actual_c, expected_c, size) ? "true" : "false");
	printf("time spent executing by the GPU (%d threads, %d blocks): %.5f ms\n", threads.x * threads.y, blocks.x * blocks.y, static_cast<double>(gpu_time));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(gpu_a);
	cudaFree(gpu_c);
}

int main(int argc, char* argv[])
{
	constexpr auto threads_for_block = 1024;
	constexpr auto vector_len = 1024 * 1024 * 128;
	constexpr auto matrix_len = 4;
	constexpr auto vector_size = static_cast<int>(vector_len) * sizeof(int);
	const auto a = static_cast<int*>(malloc(vector_size));
	const auto b = static_cast<int*>(malloc(vector_size));
	const auto c = static_cast<int*>(malloc(vector_size));
	for (int i = 0; i < vector_len; i++)
	{
		const int k = i;
		a[k] = 255;
		b[k] = 255;
		c[k] = 65025;
	}

	multiply_vector(a, b, c, vector_len);

	free(a);
	free(b);
	free(c);

	// --------------------------------------------------------------------------------------------------

	constexpr auto matrix_size = static_cast<unsigned long long>(static_cast<int>(matrix_len)) * matrix_len * sizeof(int);
	const auto am = static_cast<int*>(malloc(matrix_size));
	const auto cm = static_cast<int*>(malloc(matrix_size));
	for (int i = 0; i < matrix_len; i++)
		for (int j = 0; j < matrix_len; j++)
		{
			const int k = matrix_len * i + j;
			am[k] = i;
			cm[k] = j;
		}

	T_matrix(am, cm, matrix_len);

	free(am);
	free(cm);

	return EXIT_SUCCESS;
}