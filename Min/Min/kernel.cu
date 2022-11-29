
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>
#include <ctime>

constexpr auto block_size = 256;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ int min_g(const int a, const int b)
{
	if(a==0 || b==0)
	{
		return INT32_MAX;
	}

	if (a < b)
	{
		return a;
	}
	return b;
}

__global__ void reduce5(const int* in_data, int* out_data)
{
	__shared__ int data[block_size];
	const int tid = static_cast<int>(threadIdx.x);
	const int i = static_cast<int>(2 * blockIdx.x * blockDim.x + threadIdx.x);
	data[tid] = min_g(in_data[i], in_data[i + blockDim.x]);
	__syncthreads();
	for (int s = static_cast<int>(blockDim.x) / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			data[tid] = min_g(data[tid], data[tid + s]);
		__syncthreads();
	}
	if (tid < 32)
	{
		data[tid] = min_g(data[tid], data[tid + 32]);
		data[tid] = min_g(data[tid], data[tid + 16]);
		data[tid] = min_g(data[tid], data[tid + 8]);
		data[tid] = min_g(data[tid], data[tid + 4]);
		data[tid] = min_g(data[tid], data[tid + 2]);
		data[tid] = min_g(data[tid], data[tid + 1]);
	}
	if (tid == 0)
		out_data[blockIdx.x] = data[0];
}

int reduce(const int* data, const int n)
{
	const auto matrix_size = n * sizeof(int);

	int* sums = nullptr;
	int* data_cuda = nullptr;

	const int num_blocks = n / block_size;
	int res = INT32_MAX;

	cudaMalloc(reinterpret_cast<void**>(&data_cuda), matrix_size);
	cudaMalloc(reinterpret_cast<void**>(&sums), matrix_size);
	cudaMemcpy(data_cuda, data, matrix_size, cudaMemcpyHostToDevice);

	reduce5 <<< dim3(num_blocks), dim3(block_size) >>> (data_cuda, sums);
	gpuErrchk(cudaPeekAtLastError())
	gpuErrchk(cudaDeviceSynchronize())

	if (num_blocks > block_size)
	{
		res = reduce(sums, num_blocks);
	}
	else
	{
		const auto sums_host = new int[num_blocks];
		cudaMemcpy(sums_host, sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < num_blocks; i++)
		{
			res = res < sums_host[i] ? res : sums_host[i];
		}
		delete[] sums_host;
	}
	cudaFree(sums);
	return res;
}

bool check_min(const int* a, const int actual_min, const int n)
{
	auto expected_min = a[0];
	for (auto i = 0; i < n; i++)
	{
		if (expected_min > a[i])
		{
			expected_min = a[i];
		}
	}

	std::cout << "Expected min: " << expected_min << std::endl;

	return expected_min == actual_min;
}

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	constexpr auto matrix_len = 1024 * 1024;
	constexpr auto matrix_size = static_cast<int>(matrix_len) * sizeof(int);
	const auto a = static_cast<int*>(malloc(matrix_size));
	for (int i = 0; i < matrix_len; i++)
	{
		a[i] = rand() % matrix_len + block_size;
	}

	auto actual_min = reduce(a, matrix_len);
	std::cout << "Actual min_g = " << actual_min << ". Result is right: " << (check_min(a, actual_min, matrix_len) == 1 ? "true" : "false");

	free(a);

	return EXIT_SUCCESS;
}
