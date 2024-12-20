#include "function.h"
#define N 8

struct Node {
    int value;
    int weight;
    int marker;
    Node *left;
    Node *right;
    Node(int value, int weight, int marker) : value(value), weight(weight), marker(marker), left(nullptr), right(nullptr) {}
    Node(int value, int weight, Node *left, Node *right) : value(value), weight(weight), left(left), right(right) {}
};
__constant__ double luminance_quantization_table[N][N] = {
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
__constant__ double chrominance_quantization_table[N][N] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};
//zigzag indice
__constant__ int zigzagIndices[64] = {
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63
};

// __constant__ int inverseZigzagIndices[64] = {
//     0,  1,  5,  6, 14, 15, 27, 28,
//     2,  4,  7, 13, 16, 26, 29, 42,
//     3,  8, 12, 17, 25, 30, 41, 43,
//     9, 11, 18, 24, 31, 40, 44, 53,
//    10, 19, 23, 32, 39, 45, 52, 54,
//    20, 22, 33, 38, 46, 51, 55, 60,
//    21, 34, 37, 47, 50, 56, 59, 61,
//    35, 36, 48, 49, 57, 58, 62, 63
// };
__constant__ double precomputedCosine[N][N];

void precomputeCosineValues() {
    double hostCosine[N][N];
    for (int u = 0; u < N; ++u) {
        for (int x = 0; x < N; ++x) {
            hostCosine[u][x] = cos((2.0 * x + 1.0) * u * M_PI / (2.0 * N));
        }
    }
    cudaMemcpyToSymbol(precomputedCosine, hostCosine, sizeof(double) * N * N);
}

// DCT kernel optimized with pre-computed cosine values
__global__ void dctKernelOptimized(double* block, double* dctBlock) {
    __shared__ double sharedBlock[N][N];

    int u = threadIdx.x;
    int v = threadIdx.y;

    // Load the block into shared memory
    if (u < N && v < N) {
        sharedBlock[u][v] = block[u * N + v];
    }
    __syncthreads();

    // Calculate DCT using precomputed cosine values
    if (u < N && v < N) {
        double sum = 0.0;
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                double cos1 = precomputedCosine[u][x];
                double cos2 = precomputedCosine[v][y];
                sum += sharedBlock[x][y] * cos1 * cos2;
            }
        }

        double alphaU = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
        double alphaV = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
        dctBlock[u * N + v] = 0.25 * alphaU * alphaV * sum;
    }
}

// Quantization kernel
__global__ void quantizeKernel(double* dctBlock, int* quantBlock, int channel) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < N && j < N) {
        if (channel == 0) {
            quantBlock[i * N + j] = round(dctBlock[i * N + j] / luminance_quantization_table[i][j]);
        } else {
            quantBlock[i * N + j] = round(dctBlock[i * N + j] / chrominance_quantization_table[i][j]);
        }
    }
}

// Zigzag kernel
__global__ void zigzagKernel(int* quantBlock, int* zigzagBlock) {
    int idx = threadIdx.x;
    if (idx < N * N) {
        zigzagBlock[idx] = quantBlock[zigzagIndices[idx]];
    }
}

std::vector<std::tuple<int, int, int>> RunLengthEncoding(const int* zigzag) {
    std::vector<std::tuple<int, int, int>> rle;
    int count = 1;
    int marker = 0;

    for (int i = 1; i < N * N; i++) {
        if (zigzag[i] == zigzag[i - 1]) {
            count++;
        } else {
            rle.push_back({count, zigzag[i - 1], marker++});
            count = 1;
        }
    }
    rle.push_back({count, zigzag[N * N - 1], marker});
    return rle;
}
//build huffman tree
Node *BuildHuffmanTree(std::vector<std::tuple<int, int, int>> &rle) {
    std::vector<Node *> nodes;
    for (const auto &p : rle) {
        nodes.push_back(new Node(std::get<1>(p), std::get<0>(p), std::get<2>(p)));   
    }
    while (nodes.size() > 1) {
        sort(nodes.begin(), nodes.end(), [](Node *a, Node *b) { return a->weight < b->weight; });
        Node *left = nodes[0];
        Node *right = nodes[1];
        nodes.erase(nodes.begin(), nodes.begin() + 2);
        Node *parent = new Node(INT_MAX, left->weight + right->weight, 0);
        parent->left = left;
        parent->right = right;
        nodes.push_back(parent);
    }
    return nodes[0];
}

void BuildHuffmanTable(Node *root, std::vector<std::tuple<int, int, std::string>> &table, std::string code = "") {
    if (!root) {
        return;
    }
    
    if (!root->left && !root->right && root->value != INT_MAX) {
        table.push_back({root->value, root->marker, code});
        return;
    }
    
    BuildHuffmanTable(root->left, table, code + "0");
    BuildHuffmanTable(root->right, table, code + "1");
}

std::vector<bool> HuffmanEncoding(std::vector<std::tuple<int, int, int>> &rle, std::vector<std::tuple<int, int, std::string>> &huffman_table) {
    std::unordered_map<std::string, std::string> huffman_map;
    for (const auto &entry : huffman_table) {
        std::string key = std::to_string(std::get<0>(entry)) + "|" + std::to_string(std::get<1>(entry));
        huffman_map[key] = std::get<2>(entry);
    }
    
    std::vector<bool> huffman_code;
    for (const auto &p : rle) {
        std::string key = std::to_string(std::get<1>(p)) + "|" + std::to_string(std::get<2>(p));
        auto it = huffman_map.find(key);
        if (it != huffman_map.end()) {
            for (char c : it->second) {
                huffman_code.push_back(c == '1');
            }
        }
    }
    return huffman_code;
}

void CompressHuffmanTree(Node *root, std::vector<bool> &compressed_structure, 
                         std::vector<int> &compressed_values, std::vector<int> &compressed_frequencies, std::vector<int> &compressed_marker) {
    if (!root) {
        return;
    }
    if (!root->left && !root->right) {
        compressed_structure.push_back(1);
        compressed_values.push_back(root->value);
        compressed_frequencies.push_back(root->weight);
        compressed_marker.push_back(root->marker);
    } else {
        compressed_structure.push_back(0);
    }
    CompressHuffmanTree(root->left, compressed_structure, compressed_values, compressed_frequencies, compressed_marker);
    CompressHuffmanTree(root->right, compressed_structure, compressed_values, compressed_frequencies, compressed_marker);
}

Node* DecompressHuffmanTree(const std::vector<bool> &compressed_structure, 
                            const std::vector<int> &compressed_values, 
                            const std::vector<int> &compressed_frequencies, 
                            const std::vector<int> &compressed_marker,
                            int &idx_struct, int &idx_value) {
    if (static_cast<size_t>(idx_struct) >= compressed_structure.size()) {
        return nullptr;
    }

    bool is_leaf = compressed_structure[idx_struct++];
    if (is_leaf) {
        int value = compressed_values[idx_value];
        int frequency = compressed_frequencies[idx_value];
        int marker = compressed_marker[idx_value];
        idx_value++;
        Node* node = new Node(value, frequency, marker);
        return node;
    } else {
        Node* node = new Node(INT_MAX, 0, 0); // Internal node
        node->left = DecompressHuffmanTree(compressed_structure, compressed_values, compressed_frequencies, compressed_marker, idx_struct, idx_value);
        node->right = DecompressHuffmanTree(compressed_structure, compressed_values, compressed_frequencies, compressed_marker, idx_struct, idx_value);
        return node;
    }
}

std::vector<std::pair<int, int>> HuffmanDecoding(std::vector<bool> &huffman_code, Node *root) {
    std::vector<std::pair<int, int>> decoded_rle;
    Node *node = root;
    
    for (bool bit : huffman_code) {
        node = bit ? node->right : node->left;
        
        if (!node->left && !node->right) {
            decoded_rle.push_back({node->weight, node->value});
            node = root;
        }
    }
    
    return decoded_rle;
}
//decode run-length encoding
std::vector<int> DecodeRunLengthEncoding(std::vector<std::pair<int, int>> &rle) {
    std::vector<int> zigzag;
    for (const auto &p : rle) {
        for (int i = 0; i < p.first; i++) {
            zigzag.push_back(p.second);
        }
    }
    return zigzag;
}

__global__ void inverseZigzagKernel(int* zigzagBlock, int* quantBlock) {
    int idx = threadIdx.x;
    if (idx < N * N) {
        quantBlock[zigzagIndices[idx]] = zigzagBlock[idx];
    }
}

__global__ void dequantizeKernel(int* quantBlock, double* dctBlock, int channel) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < N && j < N) {
        if (channel == 0) {
            dctBlock[i * N + j] = quantBlock[i * N + j] * luminance_quantization_table[i][j];
        } else {
            dctBlock[i * N + j] = quantBlock[i * N + j] * chrominance_quantization_table[i][j];
        }
    }
    
}

__global__ void inverseDctKernelOptimized(double* dctBlock, double* block) {
    __shared__ double sharedDctBlock[N][N];

    int x = threadIdx.x;
    int y = threadIdx.y;

    // Load the DCT block into shared memory
    if (x < N && y < N) {
        sharedDctBlock[x][y] = dctBlock[x * N + y];
    }
    __syncthreads();

    // Calculate the inverse DCT using precomputed cosine values
    if (x < N && y < N) {
        double sum = 0.0;
        for (int u = 0; u < N; ++u) {
            for (int v = 0; v < N; ++v) {
                double cos1 = precomputedCosine[u][x];
                double cos2 = precomputedCosine[v][y];
                double alphaU = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
                double alphaV = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
                sum += alphaU * alphaV * sharedDctBlock[u][v] * cos1 * cos2;
            }
        }
        block[x * N + y] = 0.25 * sum;
    }
}

void DeleteHuffmanTree(Node* root) {
    if (root) {
        DeleteHuffmanTree(root->left);
        DeleteHuffmanTree(root->right);
        delete root;
    }
}

void JPEGCompress(double *&perChannel, int channel, int width, int height) {
    // Precompute cosine values for the DCT
    precomputeCosineValues();
    // Variable zone
    double h_dctBlock[N * N];
    double h_idctBlock[N * N];
    double* d_block;
    double* d_dctBlock;
    int* d_quantBlock;
    int* d_zigzagBlock;
    int* d_rleValues;
    int* d_rleCounts;
    int* d_rleLength;
    int* d_deRleValues;
    int* d_deRleCounts;
    int* d_dezigzagBlock;
    int* d_dequantBlock;
    double* d_idctBlock;
    double* d_decompressBlock;
    int* h_zigzag;
    int* h_dezigzag;
    // Encode CUDA Malloc zone with error checking
    cudaMalloc(&d_block, N * N * sizeof(double));
    cudaMalloc(&d_dctBlock, N * N * sizeof(double));
    cudaMalloc(&d_quantBlock, N * N * sizeof(int));
    cudaMalloc(&d_zigzagBlock, N * N * sizeof(int));
    cudaMalloc(&d_rleValues, N * N * sizeof(int));
    cudaMalloc(&d_rleCounts, N * N * sizeof(int));
    cudaMalloc(&d_rleLength, sizeof(int));
    // Decode CUDA Malloc zone with error checking
    cudaMalloc(&d_deRleValues, N * N * sizeof(int));
    cudaMalloc(&d_deRleCounts, N * N * sizeof(int));
    cudaMalloc(&d_dezigzagBlock, N * N * sizeof(int));
    cudaMalloc(&d_dequantBlock, N * N * sizeof(int));
    cudaMalloc(&d_idctBlock, N * N * sizeof(double));
    cudaMalloc(&d_decompressBlock, N * N * sizeof(double));
    dim3 blockDim(N, N);
    dim3 zigzagBlockDim(N * N);
    cudaError_t err; 
    for (int i = 0; i < height; i += N) {
        for (int j = 0; j < width; j += N) {
            // Copy current 8x8 block to host memory with boundary check
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    if ((i + x) < height && (j + y) < width) {
                        h_dctBlock[x * N + y] = perChannel[(i + x) * width + j + y];
                    } else {
                        h_dctBlock[x * N + y] = 0.0;  // Padding with 0 if out of bounds
                    }
                }
            }
            // Copy block to device memory
            cudaMemcpy(d_block, h_dctBlock, N * N * sizeof(double), cudaMemcpyHostToDevice);

            // Launch optimized DCT kernel
            dctKernelOptimized<<<1, blockDim>>>(d_block, d_dctBlock);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }

            // Launch Quantization kernel
            quantizeKernel<<<1, blockDim>>>(d_dctBlock, d_quantBlock, channel);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }

            // Launch Zigzag kernel
            zigzagKernel<<<1, zigzagBlockDim>>>(d_quantBlock, d_zigzagBlock);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            h_zigzag =  (int*) malloc(N * N * sizeof(int));
            cudaMemcpy(h_zigzag, d_zigzagBlock, N * N * sizeof(int), cudaMemcpyDeviceToHost);
            // Launch Run-Length Encoding kernel 
            std::vector<std::tuple<int, int, int>> rle = RunLengthEncoding(h_zigzag);
            // Huffman Tree 和 Table
            Node* huffmanTree = BuildHuffmanTree(rle);
            std::vector<std::tuple<int, int, std::string>> huffmanTable;
            BuildHuffmanTable(huffmanTree, huffmanTable, "");
            std::vector<bool> huffmanCode = HuffmanEncoding(rle, huffmanTable);
            std::vector<bool> compressedStructure;
            std::vector<int> compressedValues;
            std::vector<int> compressed_frequencies;
            std::vector<int> compressed_marker;
            CompressHuffmanTree(huffmanTree, compressedStructure, compressedValues, compressed_frequencies, compressed_marker);
            DeleteHuffmanTree(huffmanTree);
            int index = 0, valueIndex = 0;
            Node* decompressRoot = DecompressHuffmanTree(compressedStructure, compressedValues, compressed_frequencies, compressed_marker, index, valueIndex);
            std::vector<std::pair<int, int>> rle_de = HuffmanDecoding(huffmanCode, decompressRoot);
            DeleteHuffmanTree(decompressRoot);
            std::vector<int> zigzag_de = DecodeRunLengthEncoding(rle_de);
            h_dezigzag =  (int*) malloc(N * N * sizeof(int));
            for(size_t t = 0; t < N*N; t++){
                h_dezigzag[t] = zigzag_de[t];
            }
            cudaMemcpy(d_dezigzagBlock, h_dezigzag, N * N * sizeof(int), cudaMemcpyHostToDevice);
            // Inverse Zigzag
            inverseZigzagKernel<<<1, zigzagBlockDim>>>(d_dezigzagBlock, d_dequantBlock);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            // Dequantization
            dequantizeKernel<<<1, blockDim>>>(d_dequantBlock, d_idctBlock, channel);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            
            // Inverse DCT
            inverseDctKernelOptimized<<<1, blockDim>>>(d_idctBlock, d_decompressBlock);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }

            // Copy result back to host memory
            err = cudaMemcpy(h_idctBlock, d_decompressBlock, N * N * sizeof(double), cudaMemcpyDeviceToHost);
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    if ((i + x) < height && (j + y) < width) {
                        perChannel[(i + x) * width + j + y] = h_idctBlock[x * N + y];
                    }
                }
            }
        }
    }

    // Free device memory
    cudaFree(d_block);
    cudaFree(d_dctBlock);
    cudaFree(d_quantBlock);
    cudaFree(d_zigzagBlock);
    cudaFree(d_rleValues);
    cudaFree(d_rleCounts);
    cudaFree(d_rleLength);
    cudaFree(d_deRleValues);
    cudaFree(d_deRleCounts);
    cudaFree(d_dezigzagBlock);
    cudaFree(d_dequantBlock);
    cudaFree(d_idctBlock);
    cudaFree(d_decompressBlock);
    free(h_dezigzag);
    free(h_zigzag);
}



