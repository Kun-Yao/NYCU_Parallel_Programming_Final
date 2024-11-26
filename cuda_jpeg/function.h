#ifndef FUNCTION_H
#define FUNCTION_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

#define MAX_THREADS_PER_BLOCK 1024
void convertRGBToYCbCr(const unsigned char *img, double **yCbCr, int width, int height);
void convertYCbCrToRGB(double **yCbCr, unsigned char *img, int width, int height);
void JPEGCompress(double *&perChannel, int channel, int width, int height);

#endif
