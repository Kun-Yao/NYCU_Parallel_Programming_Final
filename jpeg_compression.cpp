#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
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
                           cos((2 * x + 1) * u * CV_PI / (2 * N)) *
                           cos((2 * y + 1) * v * CV_PI / (2 * N));
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
                           cos((2 * x + 1) * u * CV_PI / (2 * N)) *
                           cos((2 * y + 1) * v * CV_PI / (2 * N));
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
    int i = 0, j = 0;
    bool up = true;

    while (i < N && j < N) {
        zigzag.push_back(block[i * N + j]);
        if (up) {
            if (i > 0 && j < N - 1) {
                i--; j++;
            } else if (j == N - 1) {
                i++;
                up = false;
            } else {
                j++;
                up = false;
            }
        } else {
            if (j > 0 && i < N - 1) {
                i++; j--;
            } else if (i == N - 1) {
                j++;
                up = true;
            } else {
                i++;
                up = true;
            }
        }
    }
}

// 逆 ZigZag 掃描
void inverse_zigzag_scan(vector<double> &zigzag, double *block) {
    int i = 0, j = 0, k = 0;
    bool up = true;

    while (i < N && j < N) {
        block[i * N + j] = zigzag[k++];
        if (up) {
            if (i > 0 && j < N - 1) {
                i--; j++;
            } else if (j == N - 1) {
                i++;
                up = false;
            } else {
                j++;
                up = false;
            }
        } else {
            if (j > 0 && i < N - 1) {
                i++; j--;
            } else if (i == N - 1) {
                j++;
                up = true;
            } else {
                i++;
                up = true;
            }
        }
    }
}

// JPEG 壓縮過程
void JPEG(Mat &img, Mat &decompressed_img, int channel) {
    int rows = img.rows;
    int cols = img.cols;
    decompressed_img = Mat::zeros(rows, cols, CV_64F);

    for (int i = 0; i < rows; i += N) {
        for (int j = 0; j < cols; j += N) {
            double block[N * N], dct_result[N * N], q_block[N * N];
            double dq_block[N * N], idct_result[N * N];
            vector<double> zigzag;

            // 提取 8x8 塊
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    block[x * N + y] = img.at<double>(i + x, j + y);
                }
            }

            // DCT
            DCT(block, dct_result, N);

            // 量化
            quantization(dct_result, q_block, channel);

            // ZigZag 掃描
            zigzag_scan(q_block, zigzag);

            // 逆 ZigZag 掃描
            inverse_zigzag_scan(zigzag, dq_block);

            // 去量化
            dequantization(dq_block, dq_block, channel);

            // IDCT
            IDCT(dq_block, idct_result, N);

            // 寫回解壓縮影像
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    decompressed_img.at<double>(i + x, j + y) = idct_result[x * N + y];
                }
            }
        }
    }
}

int main() {
    // 載入影像
    Mat lena = imread("lena.png", IMREAD_GRAYSCALE);
    lena.convertTo(lena, CV_64F);

    // JPEG 壓縮
    Mat decompressed_img;
    JPEG(lena, decompressed_img, 0);

    // 計算 PSNR
    double psnr_val = PSNR(lena, decompressed_img);
    cout << "PSNR: " << psnr_val << endl;

    // 儲存解壓縮影像
    decompressed_img.convertTo(decompressed_img, CV_8U);
    imwrite("decompressed_lena.jpg", decompressed_img);

    return 0;
}
