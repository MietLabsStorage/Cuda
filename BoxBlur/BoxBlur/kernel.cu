#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "helper_image.h"
#include "cuda_runtime_api.h"

using namespace std;

texture<unsigned char, 2, cudaReadModeNormalizedFloat> g_BoxBlur;
__global__
void BoxBlur_kernel(unsigned char* pDst, float radius, int w, int h, int p)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx < w && tidy < h)
    {
        float r = 0;
        for (int ir = -radius; ir <= radius; ir++)
            for (int ic = -radius; ic <= radius; ic++)
            {
                r += tex2D(g_BoxBlur, tidx + 0.5f + ic, tidy + 0.5f + ir);
            }
        r /= ((2 * radius + 1) * (2 * radius + 1));
        pDst[tidx + tidy * p] = (unsigned char)r;
    }
}

void loadImage(char* file, unsigned char** pixels, unsigned int* width, unsigned int* height)
{
    size_t file_length = strlen(file);
    if (!strcmp(&file[file_length - 3], "pgm"))
    {
        if (sdkLoadPGM<unsigned char>(file, pixels, width, height) != true)
        {
            printf("Failed to load PGM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
}
void saveImage(char* file, unsigned char* pixels, unsigned int width, unsigned int height)
{
    size_t file_length = strlen(file);
    if (!strcmp(&file[file_length - 3], "pgm"))
    {
        sdkSavePGM(file, pixels, width, height);
    }
}

unsigned int width = 2048, height = 2048;
int main()
{
    unsigned char* d_result_pixels;
    unsigned char* h_result_pixels;
    unsigned char* h_pixels = NULL;
    unsigned char* d_pixels = NULL;

    char* src_path = "mj.png";
    char* d_result_path = "mj_d.pgm";
    loadImage(src_path, &h_pixels, &width, &height);
    int image_size = sizeof(unsigned char) * width * height;
    h_result_pixels = (unsigned char*)malloc(image_size);
    cudaMalloc((void**)&d_pixels, image_size);
    cudaMalloc((void**)&d_result_pixels, image_size);
    cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice);
    int n = 16;
    dim3 block(n, n);
    dim3 grid(width / n, height / n);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar1>();
    size_t offset = 0;
    cudaError_t error = cudaBindTexture2D(0, &g_BoxBlur, d_pixels, &desc, width, height, width
        * sizeof(unsigned char));
    if (cudaSuccess != error) {
        printf("ERROR: Failed to bind texture.\n");
        exit(-1);
    }
    else {
        printf("Texture was successfully binded\n");
    }
    /* CUDA method */
    BoxBlur_kernel << < grid, block >> > (d_result_pixels, 5, width, height, 4);
    cudaMemcpy(h_result_pixels, d_result_pixels, image_size, cudaMemcpyDeviceToHost);
    saveImage(d_result_path, h_result_pixels, width, height);
    cudaUnbindTexture(&g_BoxBlur);

    printf("DONE\n");

    return 0;
}