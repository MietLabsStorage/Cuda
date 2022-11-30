#include <cassert>
#include <cstring>
#include <random>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;
typedef unsigned char uchar;
constexpr auto n = (1024 * 2); 
constexpr auto log2_warp_size = 5;
constexpr auto warp_size = 32; 
constexpr auto tag_mask = 0x07FFFFFFU; 
constexpr auto num_bins = 256; 
constexpr auto num_warps = 6; 
constexpr auto merge_threadblock_size = 256;

inline __device__ void add_byte(volatile uint* warp_hist, const uint data, uint thread_tag)
{
	uint count;
	do
	{
		count = warp_hist[data] & tag_mask;
		count = thread_tag | (count + 1);
		warp_hist[data] = count;
	} while (warp_hist[data] != count);
}

inline __device__ void add_word(volatile uint* warp_hist, const uint data, const uint tag)
{
	add_byte(warp_hist, (data >> 0) & 0xFFU, tag);
	add_byte(warp_hist, (data >> 8) & 0xFFU, tag);
	add_byte(warp_hist, (data >> 16) & 0xFFU, tag);
	add_byte(warp_hist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram_kernel(uint* partial_histograms, const uint* data, const uint data_count)
{
	__shared__ uint hist[num_bins * num_warps];
	uint* warpHist = hist + (threadIdx.x >> log2_warp_size) * num_bins;

#pragma unroll
	for (uint i = 0; i < num_bins / warp_size; i++)
		hist[threadIdx.x + i * num_warps * warp_size] = 0;

	uint tag = threadIdx.x << (32 - log2_warp_size);
	__syncthreads();


	for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < data_count;
		pos += blockDim.x * gridDim.x)
	{
		uint d = data[pos];
		add_word(warpHist, d, tag);
	}
	__syncthreads();


	for (uint bin = threadIdx.x; bin < num_bins; bin += num_warps * warp_size)
	{
		uint sum = 0;
		for (uint i = 0; i < num_warps; i++)
			sum += hist[bin + i * num_bins] & tag_mask;
		partial_histograms[blockIdx.x * num_bins + bin] = sum;
	}
}

__global__ void merge_histogram_kernel(uint* out_histogram, const uint* partial_histograms, const uint histogram_count)
{
	uint sum = 0;
	for (uint i = threadIdx.x; i < histogram_count; i += 256)
		sum += partial_histograms[blockIdx.x + i * num_bins];
	__shared__ uint data[num_bins];
	data[threadIdx.x] = sum;
	for (uint stride = num_bins / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < stride)
			data[threadIdx.x] += data[threadIdx.x + stride];
	}
	if (threadIdx.x == 0)
		out_histogram[blockIdx.x] = data[0];
}

void histogram(uint* histogram, void* data_dev, const uint byteCount)
{
	assert(byteCount % 4 == 0);
	const int n = byteCount / 4;
	int numBlocks = n / (num_warps * warp_size);
	constexpr int numPartials = 240;
	uint* partialHistograms = nullptr;
	cudaMalloc((void**)&partialHistograms, numPartials * num_bins * sizeof(uint));
	histogram_kernel << <dim3(numPartials), dim3(num_warps * warp_size) >> > (
		partialHistograms, (uint*)data_dev, n);
	merge_histogram_kernel << <dim3(num_bins), dim3(256) >> > (histogram,
		partialHistograms, numPartials);
	cudaFree(partialHistograms);
}

void randomInit(uint* a, int n, uint* h)
{
	std::mt19937 gen(1607);
	std::normal_distribution<> distr(128, 32);

	for (int i = 0; i < n; i++)
	{
		const uchar b1 = static_cast<int>(distr(gen)) & 0xFF;
		const uchar b2 = static_cast<int>(distr(gen)) & 0xFF;
		const uchar b3 = static_cast<int>(distr(gen)) & 0xFF;
		const uchar b4 = static_cast<int>(distr(gen)) & 0xFF;
		a[i] = b1 | (b2 << 8) | (b3 << 16) | (b4 << 24);
		h[b1]++;
		h[b2]++;
		h[b3]++;
		h[b4]++;
	}
}
int main(int argc, char* argv[])
{
	const auto a = new uint[n];
	uint* h_dev = nullptr;
	uint* a_dev = nullptr;
	uint h[num_bins];
	uint h_host[num_bins];
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	memset(h_host, 0, sizeof(h_host));
	randomInit(a, n, h_host);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, nullptr);
	cudaMalloc(reinterpret_cast<void**>(&a_dev), n * sizeof(uint));
	cudaMalloc(reinterpret_cast<void**>(&h_dev), num_bins * sizeof(uint));
	cudaMemcpy(a_dev, a, n * sizeof(uint), cudaMemcpyHostToDevice);
	histogram(h_dev, a_dev, 4 * n);
	cudaMemcpy(h, h_dev, num_bins * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaFree(a_dev);
	cudaFree(h_dev);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Elapsed time: %.2f\n", gpu_time);
	for (int i = 0; i < num_bins; i++)
	{
		if (h[i] != h_host[i])
			printf("Diff at %d – %d, %d\n", i, h[i], h_host[i]);
	
		for (int j = 0; j < h[i]; j++)
		{
			printf("*");
		}
		printf("\n");
	}
	delete[] a;
	return 0;
}