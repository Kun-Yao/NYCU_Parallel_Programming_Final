#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include "CycleTimer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern void convertRGBToYCbCr(const unsigned char *img, double **yCbCr, int width, int height);
extern void convertYCbCrToRGB(double **yCbCr, unsigned char *img, int width, int height);
extern void JPEGCompress(double *&perChannel, int channel, int width, int height);
int main(int argc, char* argv[]) {
    std::string inputFilename;
    std::string outputFilename = "output.png";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-f" && i + 1 < argc) {
            inputFilename = argv[++i];
        } else if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            outputFilename = argv[++i];
        }
    }
    if (inputFilename.empty()) {
        std::cerr << "Error: Input file not specified. Use -f <filename> to specify the input file.\n";
        return -1;
    }

    int width, height, channels;
    unsigned char *img = stbi_load(inputFilename.c_str(), &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Error: could not load image.\n";
        return -1;
    }
    double *yCbCr[3];
    for (int i = 0; i < 3; i++) {
        yCbCr[i] = (double*)malloc(width * height * sizeof(double));
    }
    double start = CycleTimer::currentSeconds();
    convertRGBToYCbCr(img, yCbCr, width, height);
    for (int i = 0; i < 3; i++) {
        JPEGCompress(yCbCr[i], i, width, height);
    }
    unsigned char *reconstructedImg = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
    convertYCbCrToRGB(yCbCr, reconstructedImg, width, height);
    double end = CycleTimer::currentSeconds();
    std::cout << "Time: " << (end - start) * 1000 << "ms" << std::endl;
    double mse = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < 3; k++) {
                mse += pow(img[i * width * 3 + j * 3 + k] - reconstructedImg[i * width * 3 + j * 3 + k], 2);
            }
        }
    }
    mse /= (height * width * 3);
    double psnr = 10 * log10((255.0 * 255.0) / mse);
    std::cout << "PSNR: " << psnr << std::endl;
    stbi_write_png(outputFilename.c_str(), width, height, 3, reconstructedImg, width * 3);
    // Free host memory
    stbi_image_free(img);
    free(yCbCr[0]);
    free(yCbCr[1]);
    free(yCbCr[2]);
    free(reconstructedImg);

    return 0;
}