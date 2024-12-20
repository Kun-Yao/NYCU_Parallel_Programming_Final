#include "function.h"
#include<iostream>
__global__ void rgbToYCbCr(const unsigned char *img, double *y, double *cb, double *cr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && yIdx < height) {
        int idx = yIdx * width + x;
        int rgb_idx = 3 * idx;

        // 將 unsigned char 轉成 double 來計算
        double r = (double)img[rgb_idx];
        double g = (double)img[rgb_idx + 1];
        double b = (double)img[rgb_idx + 2];

        y[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[idx] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
        cr[idx] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
    }
}

__global__ void yCbCrToRGB(const double *y, const double *cb, const double *cr, unsigned char *img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && yIdx < height) {
        int idx = yIdx * width + x;
        int rgb_idx = 3 * idx;

        double Y = y[idx];
        double Cb = cb[idx] - 128.0;
        double Cr = cr[idx] - 128.0;

        // 計算 RGB 值
        double r = Y + 1.402 * Cr;
        double g = Y - 0.344136 * Cb - 0.714136 * Cr;
        double b = Y + 1.772 * Cb;
    
        // 加上 0.5 做四捨五入，並確保值在 0-255 範圍內
        img[rgb_idx] = (unsigned char)fmin(fmax(r, 0), 255);
        img[rgb_idx + 1] = (unsigned char)fmin(fmax(g, 0), 255);
        img[rgb_idx + 2] = (unsigned char)fmin(fmax(b, 0), 255);
    }
}

void convertRGBToYCbCr(const unsigned char *img, double **yCbCr, int width, int height) {
    double *y, *cb, *cr;
    double *d_y, *d_cb, *d_cr;
    unsigned char *d_img;
    size_t imgSize = width * height * 3 * sizeof(unsigned char);
    size_t channelSize = width * height * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_img, imgSize);
    cudaMalloc((void**)&d_y, channelSize);
    cudaMalloc((void**)&d_cb, channelSize);
    cudaMalloc((void**)&d_cr, channelSize);

    // Copy image data from host to device
    cudaMemcpy(d_img, img, imgSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Check max threads per block
    if (block.x * block.y > MAX_THREADS_PER_BLOCK) {
        printf("Error: block size too large\n");
        return;
    }

    // Launch kernel
    rgbToYCbCr<<<grid, block>>>(d_img, d_y, d_cb, d_cr, width, height);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(cudaGetErrorString(err));
        return;
    }
    // Allocate host memory for output channels
    y = (double*)malloc(channelSize);
    cb = (double*)malloc(channelSize);
    cr = (double*)malloc(channelSize);

    // Copy results from device to host
    cudaMemcpy(y, d_y, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(cb, d_cb, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(cr, d_cr, channelSize, cudaMemcpyDeviceToHost);

    yCbCr[0] = y;
    yCbCr[1] = cb;
    yCbCr[2] = cr;
    // Free device memory
    cudaFree(d_img);
    cudaFree(d_y);
    cudaFree(d_cb);
    cudaFree(d_cr);
}

void convertYCbCrToRGB(double **yCbCr, unsigned char *img, int width, int height) {
    double *d_y, *d_cb, *d_cr;
    unsigned char *d_img;
    size_t imgSize = width * height * 3 * sizeof(unsigned char);
    size_t channelSize = width * height * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_y, channelSize);
    cudaMalloc((void**)&d_cb, channelSize);
    cudaMalloc((void**)&d_cr, channelSize);
    cudaMalloc((void**)&d_img, imgSize);
    // Copy YCbCr data from host to device
    cudaMemcpy(d_y, yCbCr[0], channelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cb, yCbCr[1], channelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cr, yCbCr[2], channelSize, cudaMemcpyHostToDevice);
    // Define block and grid sizes
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch kernel
    yCbCrToRGB<<<grid, block>>>(d_y, d_cb, d_cr, d_img, width, height);
    cudaDeviceSynchronize();
    // Copy results from device to host
    cudaMemcpy(img, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_y);
    cudaFree(d_cb);
    cudaFree(d_cr);
    cudaFree(d_img);
}

