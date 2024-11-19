#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "../include/CycleTimer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

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

double chrominance_quantization_table[8][8] = {
    {7, 7, 8, 17, 17, 40, 42, 42},
    {7, 8, 10, 22, 21, 45, 52, 44},
    {9, 11, 19, 26, 26, 49, 56, 47},
    {14, 13, 26, 31, 40, 58, 60, 51},
    {18, 25, 40, 41, 48, 69, 69, 56},
    {24, 40, 51, 61, 60, 70, 78, 64},
    {40, 58, 60, 55, 64, 81, 104, 84},
    {51, 60, 70, 70, 78, 95, 91, 94}
};

vector<vector<int>> zigzag_order = {
    {0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
    {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
    {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
    {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
    {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
    {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
    {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
    {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}
};
// 離散餘弦轉換 (DCT)
void DCT(double *img, double *dct_result, int N) {
    double cu, cv, sum;
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            if (u == 0) cu = sqrt(2) / 2.0; else cu = 1;
            if (v == 0) cv = sqrt(2) / 2.0; else cv = 1;
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

// 逆離散餘弦轉換 (IDCT)
void IDCT(double *freq, double *idct_result, int N) {
    double cu, cv, sum;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            sum = 0.0;
            for (int u = 0; u < N; u++) {
                for (int v = 0; v < N; v++) {
                    if (u == 0) cu = sqrt(2) / 2.0; else cu = 1;
                    if (v == 0) cv = sqrt(2) / 2.0; else cv = 1;
                    sum += cu * cv * freq[u * N + v] *
                           cos((2 * x + 1) * u * M_PI / (2 * N)) *
                           cos((2 * y + 1) * v * M_PI / (2 * N));
                }
            }
            idct_result[x * N + y] = 0.25 * sum;
        }
    }
}

// 量化
void quantization(double *block, double *q_block, int channel) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (channel == 0) {
                q_block[i * N + j] = round(block[i * N + j] / luminance_quantization_table[i][j]);
            } else {
                q_block[i * N + j] = round(block[i * N + j] / chrominance_quantization_table[i][j]);
            }
        }
    }
}

// 去量化
void dequantization(double *block, double *dq_block, int channel) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (channel == 0) {
                dq_block[i * N + j] = round(block[i * N + j] * luminance_quantization_table[i][j]);
            } else {
                dq_block[i * N + j] = round(block[i * N + j] * chrominance_quantization_table[i][j]);
            }
        }
    }
}

// ZigZag 掃描
void zigzag_scan(double *block, vector<double> &zigzag) {
    for (const auto &pos : zigzag_order) {
        zigzag.push_back(block[pos[0] * N + pos[1]]);
    }
}

void run_length_encoding(vector<double> &zigzag, vector<pair<int, double>> &rle) {
    int count = 1;
    for (int i = 1; i < int(zigzag.size()); i++) {
        if (zigzag[i] == zigzag[i - 1]) {
            count++;
        } else {
            rle.push_back({count, zigzag[i - 1]});
            count = 1;
        }
    }
    rle.push_back({count, zigzag[zigzag.size() - 1]});
}

void run_length_decoding(vector<pair<int, double>> &rle, vector<double> &zigzag) {
    zigzag.clear();
    for (const auto &p : rle) {
        for (int i = 0; i < p.first; i++) {
            zigzag.push_back(p.second);
        }
    }
}

// reverse zigzag scan corresponding to zigzag scan
void inverse_zigzag_scan(vector<double> &zigzag, double *block) {
    for (int i = 0; i < N * N; i++) {
        vector<int> temp = zigzag_order[i];
        block[temp[0] * N + temp[1]] = zigzag[i];
    }
}

// JPEG 壓縮過程
void JPEG(double *img, double *decompressed_img, int width, int height, int channel) {
    int rows = height;
    int cols = width;

    for (int i = 0; i < rows; i += N) {
        for (int j = 0; j < cols; j += N) {
            double block[N * N], dct_result[N * N], q_block[N * N];
            double dq_block[N * N], idct_result[N * N];
            vector<double> zigzag;

            // 提取 8x8 塊
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
            quantization(dct_result, q_block, channel);

            // ZigZag 掃描
            zigzag_scan(q_block, zigzag);

            // Run-Length Encoding
            vector<pair<int, double>> rle;
            run_length_encoding(zigzag, rle);

            // Run-Lenght Decoding
            run_length_decoding(rle, zigzag);

            // 逆 ZigZag 掃描
            inverse_zigzag_scan(zigzag, dq_block);

            // 去量化
            dequantization(dq_block, dq_block, channel);

            // IDCT
            IDCT(dq_block, idct_result, N);

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

int getFileSize(const string &filename) {
    ifstream file(filename, ios::binary | ios::ate);  // 開啟檔案並定位至檔案結尾
    if (!file) {
        cout << "Failed to open file: " << filename << endl;
        return -1;
    }
    return file.tellg();  // 返回檔案大小
}

int main() {
    // 載入影像
    int width, height, channels;
    unsigned char *img = stbi_load("/home/312553027/NYCU_Parallel_Programming_Final/src/lena.bmp", &width, &height, &channels, 1);
    if (!img) {
        cout << "Failed to load image! Error: " << stbi_failure_reason() << endl;
        return -1;
    }
    // cout << "Image width: " << width << ", height: " << height << endl;

    // 計算原始影像檔案大小
    int original_size = getFileSize("/home/312553027/NYCU_Parallel_Programming_Final/src/lena.bmp");
    if (original_size == -1) return -1;

    // 將圖像資料轉換為雙精度格式
    vector<double> img_data(width * height);
    for (int i = 0; i < width * height; i++) {
        img_data[i] = static_cast<double>(img[i]);
    }

    // calculate start time
    double start = CycleTimer::currentSeconds();

    // JPEG 壓縮
    vector<double> decompressed_img(width * height, 0.0);
    JPEG(img_data.data(), decompressed_img.data(), width, height, 0);

    // calculate end time
    double end = CycleTimer::currentSeconds();

    // 計算 PSNR
    double mse = 0.0;
    for (int i = 0; i < width * height; i++) {
        double diff = img_data[i] - decompressed_img[i];
        mse += diff * diff;
    }
    mse /= width * height;
    double psnr = (mse == 0) ? INFINITY : 10 * log10(255 * 255 / mse);
    cout << "PSNR: " << psnr << endl;

    // output execution time
    std::cout << "Total execution time: " << (end - start) * 1000.0 << " ms" << std::endl;

    // 將解壓縮影像從雙精度轉換為 8 位無符號整數型態
    vector<unsigned char> decompressed_img_8u(width * height);
    for (int i = 0; i < width * height; i++) {
        // 限制像素值範圍至 0 到 255 之間
        decompressed_img_8u[i] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, decompressed_img[i])));
    }

    // 儲存解壓縮影像
    if (!stbi_write_jpg("decompressed_lena.jpg", width, height, 1, decompressed_img_8u.data(), 100)) {
        cout << "Failed to save decompressed image!" << endl;
        return -1;
    }

    // 計算壓縮後影像檔案大小
    int compressed_size = getFileSize("decompressed_lena.jpg");
    if (compressed_size == -1) return -1;

    // 計算壓縮率
    double compression_rate = (static_cast<double>(original_size) / static_cast<double>(compressed_size));
    cout << "Compression Rate: " << compression_rate << "x" << endl;



    stbi_image_free(img);
    return 0;
}