#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include "./include/CycleTimer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./include/stb_image_write.h"
#include <omp.h>
#include "./include/shishua-avx2.h"
#include "./include/immintrin.h"

using namespace std;

const int N = 8;
//luminance quantization table
vector<vector<int>> luminance_quantization_table = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};
vector<vector<double>> luminance_quantization_table_test = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};
//chrominance quantization table
vector<vector<int>> chrominance_quantization_table = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};
vector<vector<double>> chrominance_quantization_table_test = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};
//zigzag table
vector<pair<int,int>>table = { 
        {0,0},{0,1},{1,0},{2,0},{1,1},{0,2},{0,3},{1,2},
        {2,1},{3,0},{4,0},{3,1},{2,2},{1,3},{0,4},{0,5},
        {1,4},{2,3},{3,2},{4,1},{5,0},{6,0},{5,1},{4,2},
        {3,3},{2,4},{1,5},{0,6},{0,7},{1,6},{2,5},{3,4},
        {4,3},{5,2},{6,1},{7,0},{7,1},{6,2},{5,3},{4,4},
        {3,5},{2,6},{1,7},{2,7},{3,6},{4,5},{5,4},{6,3},
        {7,2},{7,3},{6,4},{5,5},{4,6},{3,7},{4,7},{5,6},
        {6,5},{7,4},{7,5},{6,6},{5,7},{6,7},{7,6},{7,7} };
//huffman tree node
struct Node {
    int value;
    int weight;
    Node *left;
    Node *right;
    Node(int value, int weight) : value(value), weight(weight), left(nullptr), right(nullptr) {}
    Node(int value, int weight, Node *left, Node *right) : value(value), weight(weight), left(left), right(right) {}
};
//DCT
vector<vector<double>> DCT(vector<vector<double>> &block) {
    vector<vector<double>> dct(N, vector<double>(N, 0));
    // #pragma omp parallel for
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            double sum = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    sum += block[i][j] * cos((2 * i + 1) * u * M_PI / 16) * cos((2 * j + 1) * v * M_PI / 16);
                }
            }
            double Cu = (u == 0) ? 1 / sqrt(2) : 1;
            double Cv = (v == 0) ? 1 / sqrt(2) : 1;
            dct[u][v] = 0.25 * Cu * Cv * sum;
        }
    }
    return dct;
}
//quantization
vector<vector<int>> Quantization(vector<vector<double>> &dct, int channel) {
    vector<vector<int>> quantization(N, vector<int>(N, 0));
    vector<vector<int>> quantization_table;
    if (channel == 0) {
        quantization_table = luminance_quantization_table;
    } else {
        quantization_table = chrominance_quantization_table;
    }
    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            quantization[i][j] = round(dct[i][j] / quantization_table[i][j]);
        }
    }
    return quantization;
}
// vector<vector<int>> Quantization(vector<vector<double>> &dct, int channel) {
//     // using double instead of int
//     vector<vector<double>> quantization(N, vector<double>(N, 0));
//     vector<vector<double>> quantization_table;

//     if (channel == 0) {
//         quantization_table = luminance_quantization_table_test;
//     } else {
//         quantization_table = chrominance_quantization_table_test;
//     }

//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j += 4) {
//             __m256d dct_vals = _mm256_loadu_pd(&dct[i][j]);
//             __m256d quant_vals = _mm256_loadu_pd(&quantization_table[i][j]);

//             __m256d result = _mm256_div_pd(dct_vals, quant_vals);

//             result = _mm256_round_pd(result, _MM_FROUND_TO_NEAREST_INT);
//             _mm256_storeu_pd(&quantization[i][j], result);
//         }
//     }

//     // convert to int type
//     vector<vector<int>> final_quantization(N, vector<int>(N, 0));
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             final_quantization[i][j] = static_cast<int>(round(quantization[i][j]));
//         }
//     }

//     return final_quantization;
// }
vector<int> ZigZag(vector<vector<int>> &quantization) {
    vector<int> zigzag(N * N, 0);

    // processing 8 values per time
    for (int i = 0; i < N * N; i += 8) {
        int idx[8];
        for (int k = 0; k < 8; ++k) {
            idx[k] = table[i + k].first * N + table[i + k].second;
        }

        // load 8 values
        __m256i values = _mm256_set_epi32(
            quantization[idx[7] / N][idx[7] % N],
            quantization[idx[6] / N][idx[6] % N],
            quantization[idx[5] / N][idx[5] % N],
            quantization[idx[4] / N][idx[4] % N],
            quantization[idx[3] / N][idx[3] % N],
            quantization[idx[2] / N][idx[2] % N],
            quantization[idx[1] / N][idx[1] % N],
            quantization[idx[0] / N][idx[0] % N]
        );

        // store values to zigzag
        _mm256_storeu_si256((__m256i*)&zigzag[i], values);
    }

    return zigzag;
}

//zigzag
// vector<int> ZigZag(vector<vector<int>> &quantization) {
//     vector<int> zigzag(N * N, 0);
//     // #pragma omp parallel for
//     for (int i = 0; i < N * N; i++) {
//         zigzag[i] = quantization[table[i].first][table[i].second];
//     }
//     return zigzag;
// }
//run-length encoding
vector<pair<int, int>> RunLengthEncoding(vector<int> &zigzag) {
    vector<pair<int, int>> rle;
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
    return rle;
}
//build huffman tree
Node *BuildHuffmanTree(vector<pair<int, int>> &rle) {
    vector<Node *> nodes;
    for (const auto &p : rle) {
        nodes.push_back(new Node(p.second, p.first));   
    };
    while (nodes.size() > 1) {
        sort(nodes.begin(), nodes.end(), [](Node *a, Node *b) { return a->weight > b->weight; });
        Node *left = nodes.back();
        nodes.pop_back();
        Node *right = nodes.back();
        nodes.pop_back();
        Node *parent = new Node(-1, left->weight + right->weight);
        parent->left = left;
        parent->right = right;
        nodes.push_back(parent);
    }
    return nodes[0];
}
//build huffman table
void BuildHuffmanTable(Node *root, vector<pair<int, string>> &table, string code) {
    if (!root) {
        return;
    }
    if (root->value != -1) {
        table.push_back({root->value, code});
    }
    BuildHuffmanTable(root->left, table, code + "0");
    BuildHuffmanTable(root->right, table, code + "1");
}
//huffman encoding
vector<bool> HuffmanEncoding(vector<pair<int, int>> &rle, vector<pair<int, string>> &huffman_table) {
    vector<bool> huffman_code;
    for (const auto &p : rle) {
        for (const auto &q : huffman_table) {
            if (p.second == q.first) {
                for (char c : q.second) {
                    huffman_code.push_back(c - '0');
                }
            }
        }
    }
    return huffman_code;
}
//compress huffman tree
void CompressHuffmanTree(Node *root, vector<bool> &compressed_structure, vector<int> &compressed_values) {
    if (!root) {
        return;
    }
    if (!root->left && !root->right) { // Leaf node
        compressed_structure.push_back(1);
        compressed_values.push_back(root->value);
    } else {
        compressed_structure.push_back(0);
    }
    CompressHuffmanTree(root->left, compressed_structure, compressed_values);
    CompressHuffmanTree(root->right, compressed_structure, compressed_values);
}

// Decompress Huffman tree
Node* DecompressHuffmanTree(const vector<bool> &compressed_structure, const vector<int> &compressed_values, int &idx_struct, int &idx_value) {
    if (static_cast<size_t>(idx_struct) >= compressed_structure.size()) {
        return nullptr;
    }

    bool is_leaf = compressed_structure[idx_struct++];
    if (is_leaf) {
        int value = compressed_values[idx_value++];
        Node* node = new Node(value, 0);
        return node;
    } else {
        Node* node = new Node(-1, 0); // Internal node
        node->left = DecompressHuffmanTree(compressed_structure, compressed_values, idx_struct, idx_value);
        node->right = DecompressHuffmanTree(compressed_structure, compressed_values, idx_struct, idx_value);
        return node;
    }
}

//huffman decoding
vector<pair<int, int>> HuffmanDecoding(vector<bool> &huffman_code, Node *root) {
    vector<pair<int, int>> rle;
    Node *node = root;
    for (int i = 0; i < int(huffman_code.size()); i++) {
        if (huffman_code[i]) {
            node = node->right;
        } else {
            node = node->left;
        }
        if (node->value != -1) {
            rle.push_back({node->weight, node->value});
            node = root;
        }
    }
    return rle;
}
//decode run-length encoding
vector<int> DecodeRunLengthEncoding(vector<pair<int, int>> &rle) {
    vector<int> zigzag;
    for (const auto &p : rle) {
        for (int i = 0; i < p.first; i++) {
            zigzag.push_back(p.second);
        }
    }
    return zigzag;
}
//inverse zigzag
vector<vector<int>> InverseZigZag(vector<int> &zigzag) {
    vector<vector<int>> quantization(N, vector<int>(N, 0));
    // #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        quantization[table[i].first][table[i].second] = zigzag[i];
    }
    return quantization;
}
//inverse quantization
vector<vector<double>> InverseQuantization(vector<vector<int>> &quantization, int channel) {
    vector<vector<double>> dct(N, vector<double>(N, 0));
    vector<vector<int>> quantization_table;
    if (channel == 0) {
        quantization_table = luminance_quantization_table;
    } else {
        quantization_table = chrominance_quantization_table;
    }
    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dct[i][j] = quantization[i][j] * quantization_table[i][j];
        }
    }
    // 使用 SIMD 每次處理 8 個整數
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j += 8) {
    //         // 加載量化數據和量化表數據
    //         __m256i quant_vals = _mm256_loadu_si256((__m256i*)&quantization[i][j]);
    //         __m256i table_vals = _mm256_loadu_si256((__m256i*)&quantization_table[i][j]);

    //         // 進行逐元素整數乘法
    //         __m256i result = _mm256_mullo_epi32(quant_vals, table_vals);

    //         // 將結果存回 DCT 矩陣
    //         _mm256_storeu_si256((__m256i*)&dct[i][j], result);
    //     }
    // }
    return dct;
}
//inverse DCT
vector<vector<double>> InverseDCT(vector<vector<double>> &dct) {
    vector<vector<double>> block(N, vector<double>(N, 0));
    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int u = 0; u < N; u++) {
                for (int v = 0; v < N; v++) {
                    double Cu = (u == 0) ? 1 / sqrt(2) : 1;
                    double Cv = (v == 0) ? 1 / sqrt(2) : 1;
                    sum += Cu * Cv * dct[u][v] * cos((2 * i + 1) * u * M_PI / 16) * cos((2 * j + 1) * v * M_PI / 16);
                }
            }
            block[i][j] = 0.25 * sum;
        }
    }
    return block;
}
void DeleteHuffmanTree(Node* root) {
    if (root) {
        DeleteHuffmanTree(root->left);
        DeleteHuffmanTree(root->right);
        delete root;
    }
}
vector<vector<double>> JPEGCompressAndDepression(vector<vector<vector<double>>> &yCbCr, int channel, int& transport_size) {
    int height = yCbCr.size();
    int width = yCbCr[0].size();
    //Decompression image individual channel
    vector<vector<double>> yCbCr_de(height, vector<double>(width, 0));
    // 8x8 block DCT
    int block_height = height / 8;
    int block_width = width / 8;
    #pragma omp parallel for
    for (int i = 0; i < block_height; i++) {
        for (int j = 0; j < block_width; j++) {
            vector<vector<double>> block(N, vector<double>(N, 0));
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    block[x][y] = yCbCr[i * 8 + x][j * 8 + y][channel];
                }
            }
            vector<vector<double>> dct = DCT(block);
            vector<vector<int>> quantization = Quantization(dct, channel);
            vector<int> zigzag = ZigZag(quantization);
            vector<pair<int, int>> rle = RunLengthEncoding(zigzag);
            vector<bool> compressed_structure;
            vector<int> compressed_values;  
            Node* root = BuildHuffmanTree(rle);
            CompressHuffmanTree(root, compressed_structure, compressed_values);
            vector<pair<int, string>> huffman_table;
            BuildHuffmanTable(root, huffman_table, "");
            vector<bool> huffman_code = HuffmanEncoding(rle, huffman_table);
            #pragma omp atomic
            transport_size += compressed_structure.size() + compressed_values.size() * 8 + huffman_code.size();
            int index = 0, valueIndex = 0;
            Node* decompressRoot = DecompressHuffmanTree(compressed_structure, compressed_values, index, valueIndex);
            //watch decompress tree
            vector<pair<int, string>> decompressHuffmanTable;
            BuildHuffmanTable(decompressRoot, decompressHuffmanTable, "");
            vector<pair<int, int>> rle_de = HuffmanDecoding(huffman_code, decompressRoot);
            DeleteHuffmanTree(root);
            vector<int> zigzag_de = DecodeRunLengthEncoding(rle);
            vector<vector<int>> quantization_de = InverseZigZag(zigzag_de);
            vector<vector<double>> dct_de = InverseQuantization(quantization_de, channel);
            vector<vector<double>> block_de = InverseDCT(dct_de);
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    yCbCr_de[i * 8 + x][j * 8 + y] = block_de[x][y];
                }
            }
        }
    }
    return yCbCr_de;
}

int main() {

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) { // 主執行緒執行一次
            int num_threads = omp_get_num_threads();
            cout << "Number of threads: " << num_threads << endl;
        }
    }
    // 載入影像

    int width, height, channels;
    unsigned char *img = stbi_load("src/lenna.bmp", &width, &height, &channels, 3);
    if (!img) {
        cout << "Failed to load image! Error: " << stbi_failure_reason() << endl;
        return -1;
    }
    cout << "Image width: " << width << ", height: " << height << ", channel:" << channels << endl;

    //jpeg compression
    // rgb to yCbCr
    double start = CycleTimer::currentSeconds();
    vector<vector<vector<double>>> yCbCr(height, vector<vector<double>>(width, vector<double>(3, 0)));
    // #pragma omp parallel for
    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         yCbCr[i][j][0] = 0.299 * img[i * width * 3 + j * 3] + 0.587 * img[i * width * 3 + j * 3 + 1] + 0.114 * img[i * width * 3 + j * 3 + 2];
    //         yCbCr[i][j][1] = 128 - 0.168736 * img[i * width * 3 + j * 3] - 0.331264 * img[i * width * 3 + j * 3 + 1] + 0.5 * img[i * width * 3 + j * 3 + 2];
    //         yCbCr[i][j][2] = 128 + 0.5 * img[i * width * 3 + j * 3] - 0.418688 * img[i * width * 3 + j * 3 + 1] - 0.081312 * img[i * width * 3 + j * 3 + 2];
    //     }
    // }
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width * 3 + j * 3; // 計算 RGB 的索引
            float r = img[index];
            float g = img[index + 1];
            float b = img[index + 2];

            // YCbCr color space convertion
            yCbCr[i][j][0] = 0.299f * r + 0.587f * g + 0.114f * b;                         // Y
            yCbCr[i][j][1] = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;            // Cb
            yCbCr[i][j][2] = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;            // Cr

            // range limited 0 - 255
            yCbCr[i][j][0] = fmin(fmax(yCbCr[i][j][0], 0.0f), 255.0f);
            yCbCr[i][j][1] = fmin(fmax(yCbCr[i][j][1], 0.0f), 255.0f);
            yCbCr[i][j][2] = fmin(fmax(yCbCr[i][j][2], 0.0f), 255.0f);
        }
    }

    int transport_size = 0;
    vector<vector<double>> y_de = JPEGCompressAndDepression(yCbCr, 0, transport_size);
    vector<vector<double>> Cb_de = JPEGCompressAndDepression(yCbCr, 1, transport_size);
    vector<vector<double>> Cr_de = JPEGCompressAndDepression(yCbCr, 2, transport_size);
    // Convert back to RGB
    vector<vector<vector<double>>> rgb_de(height, vector<vector<double>>(width, vector<double>(3, 0)));
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            rgb_de[i][j][0] = y_de[i][j] + 1.402 * (Cr_de[i][j] - 128);
            rgb_de[i][j][1] = y_de[i][j] - 0.344136 * (Cb_de[i][j] - 128) - 0.714136 * (Cr_de[i][j] - 128);
            rgb_de[i][j][2] = y_de[i][j] + 1.772 * (Cb_de[i][j] - 128);
            for (int k = 0; k < 3; k++) {
                rgb_de[i][j][k] = (rgb_de[i][j][k] > 255) ? 255 : (rgb_de[i][j][k] < 0) ? 0 : rgb_de[i][j][k];
            }
        }
    }

    //show image
    vector<unsigned char> img_de(width * height * 3);
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img_de[i * width * 3 + j * 3] = rgb_de[i][j][0];
            img_de[i * width * 3 + j * 3 + 1] = rgb_de[i][j][1];
            img_de[i * width * 3 + j * 3 + 2] = rgb_de[i][j][2];
        }
    }
    double end = CycleTimer::currentSeconds();
    cout << "Time: " << (end - start) * 1000 << "ms" << endl;
    //PSNR 3 channel
    double mse = 0;
    #pragma omp parallel for reduction(+:mse)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < 3; k++) {
                mse += pow(img[i * width * 3 + j * 3 + k] - img_de[i * width * 3 + j * 3 + k], 2);
            }
        }
    }
    mse /= (height * width * 3);
    double psnr = 10 * log10(255 * 255 / mse);
    cout << "PSNR: " << psnr << endl;
    //compression ratio
    double compression_ratio = (transport_size + 8*8*4*2) / double(width * height * 3 * 8);
    cout << "Compression ratio: " << compression_ratio << endl;
    stbi_write_png("lena_de.png", width, height, 3, img_de.data(), width * 3);
    stbi_image_free(img);
    return 0;

}