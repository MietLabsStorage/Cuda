#include "helper_image.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

using namespace std;

texture<unsigned char, 2, cudaReadModeElementType> g_Bilinear;
__global__ void Bilinear_kernel(unsigned char* pDst,
    float factor,
    int w, int h)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx < w && tidy < h)
    {
        float center = tidx / factor;
        int start = (int)center;
        int stop = start + 1.0f;
        float t = center - start;
        float a = tex2D(g_Bilinear, tidy + 0.5f, start + 0.5f);
        float b = tex2D(g_Bilinear, tidy + 0.5f, stop + 0.5f);
        float linear = a + t*(b-a);
        pDst[tidx + tidy * w] = (int)linear;
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

unsigned int width = 512, height = 512;

void rotate(char* name_in, char* name_out)
{
    unsigned char* d_result_pixels;
    unsigned char* h_result_pixels;
    unsigned char* h_pixels = NULL;
    unsigned char* d_pixels = NULL;

    char* src_path = name_in;//"mj.pgm";
    char* d_result_path = name_out;// "mj_d.pgm";
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
    cudaError_t error = cudaBindTexture2D(0, &g_Bilinear, d_pixels, &desc, width, height, width
        * sizeof(unsigned char));
    if (cudaSuccess != error) {
        printf("ERROR: Failed to bind texture.\n");
        exit(-1);
    }
    else {
        printf("Texture was successfully binded\n");
    }
    Bilinear_kernel << < grid, block >> > (d_result_pixels, 2, width, height );
    cudaMemcpy(h_result_pixels, d_result_pixels, image_size, cudaMemcpyDeviceToHost);

    saveImage(d_result_path, h_result_pixels, width, height);
    cudaUnbindTexture(&g_Bilinear);
}

int main()
{
    rotate("mj.pgm", "mj1.pgm");
    rotate("mj1.pgm", "mj2.pgm");
    printf("DONE\n");

    return 0;
}
