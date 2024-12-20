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
   0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

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
__global__ void dctKernelOptimized(double* input, double* output, int totalBlocks, int width, int height) {
    __shared__ double sharedBlock[N][N];
    int blocku = blockIdx.x;
    int blockv = blockIdx.y;
    int u = threadIdx.x;
    int v = threadIdx.y;
    // Boundary check
    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        sharedBlock[u][v] = input[blocku * width * N + u * width + blockv * N + v ];
    } else {
        sharedBlock[u][v] = 0.0; // Padding
    }
    __syncthreads();

    double sum = 0.0;
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            sum += sharedBlock[x][y] * precomputedCosine[u][x] * precomputedCosine[v][y];
        }
    }
    double alphaU = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
    double alphaV = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        output[blocku * width * N + u * width + blockv * N + v] = 0.25 * alphaU * alphaV * sum;
    }
}

__global__ void quantizeKernel(double* dctBlock, int* quantBlock, int totalBlocks, int channel, int width, int height) {
    int blocku = blockIdx.x;
    int blockv = blockIdx.y;
    int u = threadIdx.x;
    int v = threadIdx.y;

    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        double quantValue = (channel == 0) ? luminance_quantization_table[u][v] : chrominance_quantization_table[u][v];
        quantBlock[blocku * width * N + u * width + blockv * N + v] = round(dctBlock[blocku * width * N + u * width + blockv * N + v] / quantValue);
    }
}

__global__ void zigzagKernel(int* quantizedBlock, int* zigzagBlock, int totalBlocks, int width, int height) {
    int blocku = blockIdx.x; 
    int blockv = blockIdx.y; 
    int u = threadIdx.x;     
    int v = threadIdx.y;     

    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        int quantizedIndex = blocku * width * N + u * width + blockv * N + v;

        int zigzagPosition = zigzagIndices[u * N + v];

        zigzagBlock[blocku * width * N + blockv * N * N + zigzagPosition] = quantizedBlock[quantizedIndex];
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

__global__ void inverseZigzagKernel(int* zigzagBlock, int* originalBlock, int totalBlocks, int width, int height) {
    int blocku = blockIdx.x;
    int blockv = blockIdx.y;
    int u = threadIdx.x;
    int v = threadIdx.y;


    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {

        int originalIndex = blocku * width * N + u * width + blockv * N + v;

        int zigzagPosition = zigzagIndices[u * N + v];

        originalBlock[originalIndex] = zigzagBlock[blocku * width * N + blockv * N * N + zigzagPosition];
    }
}

__global__ void dequantizeKernel(int* quantBlock, double* dctBlock, int totalBlocks, int channel, int width, int height) {
    int blocku = blockIdx.x;
    int blockv = blockIdx.y;
    int u = threadIdx.x;
    int v = threadIdx.y;

    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        double quantValue = (channel == 0) ? luminance_quantization_table[u][v] : chrominance_quantization_table[u][v];
        dctBlock[blocku * width * N + u * width + blockv * N + v] = round(quantBlock[blocku * width * N + u * width + blockv * N + v] * quantValue);
    }
}

__global__ void idctKernelOptimized(double* input, double* output, int totalBlocks, int width, int height) {
    __shared__ double sharedBlock[N][N];
    int blocku = blockIdx.x;
    int blockv = blockIdx.y;
    int u = threadIdx.x;
    int v = threadIdx.y;

    // Boundary check
    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        sharedBlock[u][v] = input[blocku * width * N + u * width + blockv * N + v];
    } else {
        sharedBlock[u][v] = 0.0; // Padding
    }
    __syncthreads();


    double sum = 0.0;
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            double alphaX = (x == 0) ? 1.0 / sqrt(2.0) : 1.0;
            double alphaY = (y == 0) ? 1.0 / sqrt(2.0) : 1.0;
            sum += alphaX * alphaY * sharedBlock[x][y] * precomputedCosine[x][u] * precomputedCosine[y][v];
        }
    }

    if (blockIdx.y + gridDim.y * blockIdx.x < totalBlocks && u * N + v < N * N) {
        output[blocku * width * N + u * width + blockv * N + v] = 0.25 * sum;
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
    int blocksX = (height + N - 1) / N;
    int blocksY = (width + N - 1) / N;
    int totalBlocks = blocksX * blocksY;

    double *d_input, *d_dctBlocks, *d_idctBlocks, *d_outputImage;
    int *d_quantBlocks, *d_zigzagBlocks, *d_dequantBlocks, *d_dezigzagBlocks;

    cudaMalloc(&d_input, width * height * sizeof(double));
    cudaMalloc(&d_dctBlocks, totalBlocks * N * N * sizeof(double));
    cudaMalloc(&d_quantBlocks, totalBlocks * N * N * sizeof(int));
    cudaMalloc(&d_zigzagBlocks, totalBlocks * N * N * sizeof(int));
    cudaMalloc(&d_idctBlocks, totalBlocks * N * N * sizeof(double));
    cudaMalloc(&d_dequantBlocks, totalBlocks * N * N * sizeof(int));
    cudaMalloc(&d_dezigzagBlocks, totalBlocks * N * N * sizeof(int));
    cudaMalloc(&d_outputImage, width * height * sizeof(double));

    cudaMemcpy(d_input, perChannel, width * height * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(N, N);
    dim3 gridDim(blocksX, blocksY);

    dctKernelOptimized<<<gridDim, blockDim>>>(d_input, d_dctBlocks, totalBlocks, width, height);
    cudaDeviceSynchronize();

    quantizeKernel<<<gridDim, blockDim>>>(d_dctBlocks, d_quantBlocks, totalBlocks, channel, width, height);
    cudaDeviceSynchronize();

    zigzagKernel<<<gridDim, blockDim>>>(d_quantBlocks, d_zigzagBlocks, totalBlocks, width, height);
    cudaDeviceSynchronize();

    int* h_zigzagBlocks = new int[totalBlocks * N * N];
    int* h_dezigzagBlocks = new int[totalBlocks * N * N];
    int* blockZigzag = new int[ N * N];
    cudaError_t err = cudaMemcpy(h_zigzagBlocks, d_zigzagBlocks, totalBlocks * N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    for (int blockIdx = 0; blockIdx < totalBlocks; ++blockIdx) {
        blockZigzag = h_zigzagBlocks + blockIdx * N * N;
        // Run-Length Encoding
        std::vector<std::tuple<int, int, int>> rle = RunLengthEncoding(blockZigzag);
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
        for (size_t i = 0; i < N * N; ++i) {
            h_dezigzagBlocks[blockIdx * N * N + i] = zigzag_de[i];
        }
    }
    cudaMemcpy(d_dezigzagBlocks, h_dezigzagBlocks, totalBlocks * N * N * sizeof(int), cudaMemcpyHostToDevice);

    inverseZigzagKernel<<<gridDim, blockDim>>>(d_dezigzagBlocks, d_dequantBlocks, totalBlocks, width, height);
    cudaDeviceSynchronize();

    dequantizeKernel<<<gridDim, blockDim>>>(d_dequantBlocks, d_idctBlocks, totalBlocks, channel, width, height);
    cudaDeviceSynchronize();

    idctKernelOptimized<<<gridDim, blockDim>>>(d_idctBlocks, d_outputImage, totalBlocks, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(perChannel, d_outputImage, width * height * sizeof(double), cudaMemcpyDeviceToHost);

    delete[] h_zigzagBlocks;
    delete[] h_dezigzagBlocks;
    cudaFree(d_input);
    cudaFree(d_dctBlocks);
    cudaFree(d_quantBlocks);
    cudaFree(d_zigzagBlocks);
    cudaFree(d_idctBlocks);
    cudaFree(d_dequantBlocks);
    cudaFree(d_dezigzagBlocks);
    cudaFree(d_outputImage);
    cudaDeviceSynchronize();
}



