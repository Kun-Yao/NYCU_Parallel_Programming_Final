#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <getopt.h>
#include "./include/CycleTimer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./include/stb_image_write.h"

using namespace std;

const int N = 8;

void usage(const char *progname)
{
   printf("Usage: %s [options]\n", progname);
   printf("Program Options:\n");
   printf("  -f  --input   <String> Input image\n");
   printf("  -o  --output  <String> Output image\n");
}
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
    int marker;
    Node *left;
    Node *right;
    Node(int value, int weight, int marker) : value(value), weight(weight), marker(marker), left(nullptr), right(nullptr) {}
    Node(int value, int weight, Node *left, Node *right) : value(value), weight(weight), left(left), right(right) {}
};
//DCT
vector<vector<double>> DCT(vector<vector<double>> &block) {
    vector<vector<double>> dct(N, vector<double>(N, 0));
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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            quantization[i][j] = round(dct[i][j] / quantization_table[i][j]);
        }
    }
    return quantization;
}
//zigzag
vector<int> ZigZag(vector<vector<int>> &quantization) {
    vector<int> zigzag(N * N, 0);
    for (int i = 0; i < N * N; i++) {
        zigzag[i] = quantization[table[i].first][table[i].second];
    }
    return zigzag;
}
//run-length encoding
vector<tuple<int, int, int>> RunLengthEncoding(const vector<int> &zigzag) {
    vector<tuple<int, int, int>> rle;
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
Node *BuildHuffmanTree(vector<tuple<int, int, int>> &rle) {
    vector<Node *> nodes;
    for (const auto &p : rle) {
        nodes.push_back(new Node(get<1>(p), get<0>(p), get<2>(p)));   
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
//build huffman table
void BuildHuffmanTable(Node *root, vector<tuple<int, int, string>> &table, string code = "") {
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

//huffman encoding
vector<bool> HuffmanEncoding(vector<tuple<int, int, int>> &rle, vector<tuple<int, int, string>> &huffman_table) {
    unordered_map<string, string> huffman_map;
    for (const auto &entry : huffman_table) {
        string key = to_string(get<0>(entry)) + "|" + to_string(get<1>(entry));
        huffman_map[key] = get<2>(entry);
    }
    
    vector<bool> huffman_code;
    for (const auto &p : rle) {
        string key = to_string(get<1>(p)) + "|" + to_string(get<2>(p));
        auto it = huffman_map.find(key);
        if (it != huffman_map.end()) {
            for (char c : it->second) {
                huffman_code.push_back(c == '1');
            }
        }
    }
    return huffman_code;
}

//compress huffman tree
void CompressHuffmanTree(Node *root, vector<bool> &compressed_structure, 
                         vector<int> &compressed_values, vector<int> &compressed_frequencies, vector<int> &compressed_marker) {
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
// Decompress Huffman tree
Node* DecompressHuffmanTree(const vector<bool> &compressed_structure, 
                            const vector<int> &compressed_values, 
                            const vector<int> &compressed_frequencies, 
                            const vector<int> &compressed_marker,
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


//huffman decoding
vector<pair<int, int>> HuffmanDecoding(vector<bool> &huffman_code, Node *root) {
    vector<pair<int, int>> decoded_rle;
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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dct[i][j] = quantization[i][j] * quantization_table[i][j];
        }
    }
    return dct;
}
//inverse DCT
vector<vector<double>> InverseDCT(vector<vector<double>> &dct) {
    vector<vector<double>> block(N, vector<double>(N, 0));
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

struct jpeg_arg{
    int start_block_row;
    int end_block_row;
    int block_width;
    int channel;
    int* transport_size;
    vector<vector<vector<double>>>* yCbCr;
    vector<vector<double>>* channel_de;
};

void* JPEG_thread(void* arg){
    jpeg_arg* data = (jpeg_arg*)arg;
    for (int i = data->start_block_row; i < data->end_block_row; i++) {
        for (int j = 0; j < data->block_width; j++) {
            vector<vector<double>> block(N, vector<double>(N, 0));
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    block[x][y] = (*data->yCbCr)[i * 8 + x][j * 8 + y][data->channel];
                }
            }
            vector<vector<double>> dct = DCT(block);
            vector<vector<int>> quantization = Quantization(dct, data->channel);
            vector<int> zigzag = ZigZag(quantization);
            vector<tuple<int, int, int>> rle = RunLengthEncoding(zigzag);
            // Huffman Tree 和 Table
            Node* huffmanTree = BuildHuffmanTree(rle);
            vector<tuple<int, int, string>> huffmanTable;
            BuildHuffmanTable(huffmanTree, huffmanTable, "");
            vector<bool> huffman_code = HuffmanEncoding(rle, huffmanTable);
            vector<bool> compressedStructure;
            vector<int> compressedValues;
            vector<int> compressed_frequencies;
            vector<int> compressed_marker;
            CompressHuffmanTree(huffmanTree, compressedStructure, compressedValues, compressed_frequencies, compressed_marker);
            DeleteHuffmanTree(huffmanTree);
            (*data->transport_size) += compressedStructure.size() + compressedValues.size() * 8 + huffman_code.size() + compressed_frequencies.size() * 8 + compressed_marker.size() * 8;
            int index = 0, valueIndex = 0;
            Node* decompressRoot = DecompressHuffmanTree(compressedStructure, compressedValues, compressed_frequencies, compressed_marker, index, valueIndex);
            vector<pair<int, int>> rle_de = HuffmanDecoding(huffman_code, decompressRoot);
            vector<int> zigzag_de = DecodeRunLengthEncoding(rle_de);
            vector<vector<int>> quantization_de = InverseZigZag(zigzag_de);
            vector<vector<double>> dct_de = InverseQuantization(quantization_de, data->channel);
            vector<vector<double>> block_de = InverseDCT(dct_de);
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    (*data->channel_de)[i * 8 + x][j * 8 + y] = block_de[x][y];
                }
            }
        }
    }
    return nullptr;
}

vector<vector<double>> JPEGCompressAndDepression(vector<vector<vector<double>>> &yCbCr, int channel, int& transport_size, int num_thread) {
    int height = yCbCr.size();
    int width = yCbCr[0].size();
    //Decompression image individual channel
    vector<vector<double>> channel_de(height, vector<double>(width, 0));
    // 8x8 block DCT
    pthread_t threads[num_thread];
    jpeg_arg jpeg_args[num_thread];

    int block_height = height / 8;
    int block_width = width / 8;
    int thread_block_height = block_height / num_thread;
    int extra_block_height = block_height % num_thread;
    int start = 0, end;
    for (int i = 0; i < num_thread; i++){
        end = (i < extra_block_height) ? start + thread_block_height + 1 : start + thread_block_height;
        jpeg_args[i].start_block_row = start;
        jpeg_args[i].end_block_row = end;
        jpeg_args[i].block_width = block_width;
        jpeg_args[i].channel = channel;
        jpeg_args[i].transport_size = &transport_size;
        jpeg_args[i].yCbCr = &yCbCr;
        jpeg_args[i].channel_de = &channel_de;

        start = end;
        if (pthread_create(&threads[i], nullptr, JPEG_thread, (void*)&jpeg_args[i])){
            cerr << "Error creating thread" << i+1 << endl;
            return {};
        }
    }

    for (int i = 0; i < num_thread; i++){
        void* retval;
        if (pthread_join(threads[i], &retval)){
            cerr << "Error joining thread" << endl;
            return {};
        }
    }

    return channel_de;
}

struct yCbCr_arg{
    int start_row;
    int end_row;
    int width;
    unsigned char *img;
    vector<vector<vector<double>>>* yCbCr;
};

void* convert_to_yCbCr(void* arg) {
    yCbCr_arg *data = (yCbCr_arg *)arg;
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->width; j++) {
            (*data->yCbCr)[i][j][0] = 0.299 * data->img[i * data->width * 3 + j * 3] +
                                   0.587 * data->img[i * data->width * 3 + j * 3 + 1] +
                                   0.114 * data->img[i * data->width * 3 + j * 3 + 2];
            (*data->yCbCr)[i][j][1] = 128 - 0.168736 * data->img[i * data->width * 3 + j * 3] -
                                   0.331264 * data->img[i * data->width * 3 + j * 3 + 1] +
                                   0.5 * data->img[i * data->width * 3 + j * 3 + 2];
            (*data->yCbCr)[i][j][2] = 128 + 0.5 * data->img[i * data->width * 3 + j * 3] -
                                   0.418688 * data->img[i * data->width * 3 + j * 3 + 1] -
                                   0.081312 * data->img[i * data->width * 3 + j * 3 + 2];
        }
    }
    return nullptr;
}

struct rgb_arg{
    int start_row;
    int end_row;
    int width;
    vector<vector<double>>* y_de;
    vector<vector<double>>* cb_de;
    vector<vector<double>>* cr_de;
    vector<vector<vector<double>>>* RGB_de;
};
void* convert_to_rgb(void* arg) {
    rgb_arg *data = (rgb_arg *)arg;
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->width; j++) {
            (*data->RGB_de)[i][j][0] = (*data->y_de)[i][j] + 1.402 * ((*data->cr_de)[i][j] - 128);
            (*data->RGB_de)[i][j][1] = (*data->y_de)[i][j] - 0.344136 * ((*data->cb_de)[i][j] - 128) - 0.714136 * ((*data->cr_de)[i][j] - 128);
            (*data->RGB_de)[i][j][2] = (*data->y_de)[i][j] + 1.772 * ((*data->cb_de)[i][j] - 128);
            for (int k = 0; k < 3; k++) {
                (*data->RGB_de)[i][j][k] = ((*data->RGB_de)[i][j][k] > 255) ? 255 : ((*data->RGB_de)[i][j][k] < 0) ? 0 : (*data->RGB_de)[i][j][k];
            }
        }
    }
    return nullptr;
}

int main(int argc, char *argv[]) {
    
    int width, height, channels;
    const int num_thread = atoi(argv[1]);
    const char* inputFile = "sample.bmp";
    const char* outputFile = "output.png";
    // parse commandline options
    int opt;
    static struct option long_options[] = {
        {"input", 1, 0, 'f'},
        {"output", 1, 0, 'o'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "f:o:", long_options, NULL)) != EOF)
    {
        switch (opt)
        {
        case 'f':
            inputFile = optarg;
            break;
        
        case 'o':
            outputFile = optarg;
            break;

        default:
            fprintf(stderr, "Usage: %s -i <input_file> or --input=<input_file>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    // end parsing of commandline options
    // 載入影像
    unsigned char *img = stbi_load(inputFile, &width, &height, &channels, 3);
    if (!img) {
        cout << "Failed to load image! Error: " << stbi_failure_reason() << endl;
        return -1;
    }
    cout << "Image width: " << width << ", height: " << height << ", channel:" << channels << endl;

    //jpeg compression
    // rgb to yCbCr
    double start_time = CycleTimer::currentSeconds();
    vector<vector<vector<double>>> yCbCr(height, vector<vector<double>>(width, vector<double>(3, 0)));
    pthread_t threads[num_thread];
    yCbCr_arg color_args[num_thread];
    
    int thread_height = height / num_thread;
    int extra_height = height % num_thread;
    int start = 0, end;
    for (int i = 0; i < num_thread; i++){
        end = (i < extra_height) ? start + thread_height + 1 : start + thread_height;
        color_args[i].start_row = start;
        color_args[i].end_row = end;
        color_args[i].width = width;
        color_args[i].img = img;
        color_args[i].yCbCr = &yCbCr;

        start = end;
        if (pthread_create(&threads[i], NULL, convert_to_yCbCr, (void *)&color_args[i])) {
            cerr << "Error creating thread" << i+1 << endl;
            return 1;
        }
    }

    for (int i = 0; i < num_thread; i++){
        void* retval;
        if (pthread_join(threads[i], &retval)){
            cerr << "Error joining thread" << endl;
            return 2;
        }
    }

    int transport_size = 0;
    vector<vector<double>> y_de = JPEGCompressAndDepression(yCbCr, 0, transport_size, num_thread);
    vector<vector<double>> Cb_de = JPEGCompressAndDepression(yCbCr, 1, transport_size, num_thread);
    vector<vector<double>> Cr_de = JPEGCompressAndDepression(yCbCr, 2, transport_size, num_thread);
    // Convert back to RGB
    vector<vector<vector<double>>> rgb_de(height, vector<vector<double>>(width, vector<double>(3, 0)));

    rgb_arg rgb_args[num_thread];
    start = 0;
    for (int i = 0; i < num_thread; i++){
        end = (i < extra_height) ? start + thread_height + 1 : start + thread_height;
        rgb_args[i].start_row = start;
        rgb_args[i].end_row = end;
        rgb_args[i].width = width;
        rgb_args[i].RGB_de = &rgb_de;
        rgb_args[i].y_de = &y_de;
        rgb_args[i].cb_de = &Cb_de;
        rgb_args[i].cr_de = &Cr_de;

        start = end;
        if (pthread_create(&threads[i], NULL, convert_to_rgb, (void *)&rgb_args[i])) {
            cerr << "Error creating thread" << i+1 << endl;
            return 1;
        }
    }

    for (int i = 0; i < num_thread; i++){
        pthread_join(threads[i], NULL);
    }

    //show image
    vector<unsigned char> img_de(width * height * 3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img_de[i * width * 3 + j * 3] = rgb_de[i][j][0];
            img_de[i * width * 3 + j * 3 + 1] = rgb_de[i][j][1];
            img_de[i * width * 3 + j * 3 + 2] = rgb_de[i][j][2];
        }
    }
    double end_time = CycleTimer::currentSeconds();
    cout << "Time: " << (end_time - start_time) * 1000 << "ms" << endl;
    //PSNR 3 channel
    double mse = 0;
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
    stbi_write_png(outputFile, width, height, 3, img_de.data(), width * 3);
    cout << "save as " << outputFile << endl;
    stbi_image_free(img);
    return 0;

}
