#include <iostream>
#include <vector>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

using namespace std;

const int N = 8;

double luminance_quantization_table[8][8] = {
    {6, 4, 4, 6, 10, 16, 20, 24},
    {5, 5, 6, 8, 10, 23, 24, 22},
    {6, 5, 6, 10, 16, 23, 28, 22},
    {6, 7, 9, 12, 20, 35, 32, 25},
    {7, 9, 15, 22, 27, 44, 41, 31},
    {10, 14, 22, 26, 32, 42, 45, 37},
    {20, 26, 31, 35, 41, 48, 48, 40},
    {29, 37, 38, 39, 45, 40, 41, 40}
};

// 離散餘弦轉換 (DCT)
void DCT(double *img, double *dct_result, int N) {
    double cu, cv, sum;
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            cu = (u == 0) ? sqrt(2) / 2.0 : 1.0;
            cv = (v == 0) ? sqrt(2) / 2.0 : 1.0;
            sum = 0.0;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    sum += img[x * N + y] *
                           cos((2 * x + 1) * u * M_PI / (2 * N)) *
                           cos((2 * y + 1) * v * M_PI / (2 * N));
                }
            }
            dct_result[u * N + v] = 0.25 * cu * cv * sum;
        }
    }
}

// JPEG 壓縮過程
void JPEG(double *img, double *decompressed_img, int width, int height, int channel) {
    int rows = height;
    int cols = width;

    // 逐個區塊進行 JPEG 壓縮和解壓縮
    for (int i = 0; i < rows; i += N) {
        for (int j = 0; j < cols; j += N) {
            double block[N * N], dct_result[N * N], q_block[N * N];
            double dq_block[N * N], idct_result[N * N];

            // 提取 8x8 區塊
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    if (i + x < rows && j + y < cols) {
                        block[x * N + y] = img[(i + x) * cols + (j + y)];
                    }
                }
            }

            // DCT
            DCT(block, dct_result, N);

            // 量化
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    q_block[i * N + j] = round(dct_result[i * N + j] / luminance_quantization_table[i][j]);
                }
            }

            // 去量化
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    dq_block[i * N + j] = round(q_block[i * N + j] * luminance_quantization_table[i][j]);
                }
            }

            // IDCT
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    double sum = 0.0;
                    for (int u = 0; u < N; u++) {
                        for (int v = 0; v < N; v++) {
                            double cu = (u == 0) ? sqrt(2) / 2.0 : 1.0;
                            double cv = (v == 0) ? sqrt(2) / 2.0 : 1.0;
                            sum += cu * cv * dq_block[u * N + v] *
                                   cos((2 * x + 1) * u * M_PI / (2 * N)) *
                                   cos((2 * y + 1) * v * M_PI / (2 * N));
                        }
                    }
                    idct_result[x * N + y] = 0.25 * sum;
                }
            }

            // 儲存到解壓縮影像
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    if (i + x < rows && j + y < cols) {
                        decompressed_img[(i + x) * cols + (j + y)] = idct_result[x * N + y];
                    }
                }
            }
        }
    }
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("/home/312553027/NYCU_Parallel_Programming_Final/src/lena.png", &width, &height, &channels, 1);
    if (!img) {
        cout << "Failed to load image! Error: " << stbi_failure_reason() << endl;
        return -1;
    }

    // 將圖像資料轉換為雙精度格式
    vector<double> img_data(width * height);
    for (int i = 0; i < width * height; i++) {
        img_data[i] = static_cast<double>(img[i]);
    }

    // 壓縮並解壓縮圖像
    vector<double> decompressed_img(width * height, 0.0);
    JPEG(img_data.data(), decompressed_img.data(), width, height, 0);

    // 計算 PSNR
    double mse = 0.0;
    for (int i = 0; i < width * height; i++) {
        double diff = img_data[i] - decompressed_img[i];
        mse += diff * diff;
    }
    mse /= width * height;
    double psnr = (mse == 0) ? INFINITY : 10 * log10(255 * 255 / mse);
    cout << "PSNR: " << psnr << endl;

    stbi_image_free(img);
    return 0;
}
