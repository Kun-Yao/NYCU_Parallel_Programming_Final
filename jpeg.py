import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def DCT(img):
    N = img.shape[0]
    Sum = np.zeros((N, N))
    dct_result = np.zeros((N, N))

    cu = np.ones((N, 1))
    cv = np.ones((1, N))
    cu[0] = cv[0,0] = np.sqrt(2) / 2.0
    c = cu @ cv
    
    u = np.linspace(0, N-1, N).reshape((N, 1))
    v = np.linspace(0, N-1, N).reshape((1, N))
    [y, x] = np.meshgrid(u, v)
    # xx = np.cos((2 * x + 1) * v * np.pi / 6)
    # yy = np.cos((2 * y + 1) * u * np.pi / 6)
    
    Sum = np.cos((2 * y + 1) * v.T * np.pi / (2 * N)) @ img @ np.cos((2 * x + 1) * u.T * np.pi / (2 * N))
    dct_result = (2 / np.sqrt(N*N)) * c * Sum
    
    return dct_result
    
def IDCT(freq):
    N = freq.shape[0]
    cu = np.ones((N, 1))
    cv = np.ones((1, N))
    cu[0] = cv[0,0] = np.sqrt(2) / 2.0
    c = cu @ cv
    freq = freq * c
    
    x = np.linspace(0, N-1, N).reshape((N, 1))
    y = np.linspace(0, N-1, N).reshape((1, N))
    [v, u] = np.meshgrid(x, y)
    
    Sum = np.cos((2 * y.T + 1) * v * np.pi / (2 * N)) @ freq @ np.cos((2 * x.T + 1) * u * np.pi / (2 * N))
    idct_result = (2 / np.sqrt(N *N))  * Sum
    return idct_result

def quantization(dct):
    q_block = np.zeros(dct.shape)
    if channel == 0:
        q_block = np.round(dct / luminance_quantization_table)
    else:
        q_block = np.round(dct / chrominance_quantization_table)
    
    return q_block
    
    
def dequantization(inv_z):
    block = np.zeros((len(inv_z), len(inv_z[0])))
    if channel == 0:
        block = np.round(inv_z * luminance_quantization_table)
    else:
        block = np.round(inv_z * chrominance_quantization_table)
    
    return block

def zigzag_scan(q_dct):
    up = True
    zigzag = []
    height = len(q_dct)
    width = len(q_dct[0])
    
    i = 0
    j = 0
    while i < height and j < width:
        zigzag.append(q_dct[i][j])
        
        if up == True:
            if i > 0 and j < width - 1:
                i -= 1
                j += 1
            elif j == width - 1:
                up = False
                i += 1
            else:
                up = False
                j += 1
        else:
            if i < height - 1 and j > 0:
                i += 1
                j -= 1
            elif i == height - 1:
                up = True
                j += 1
            else:
                up = True
                i += 1
        
    return zigzag

def inverse_zigzag_scan(zigzag, block_size):
    up = True
    q_dct = []
    row = ['0' * 8] * block_size
    for _ in range(block_size):
        q_dct.append(row.copy())
    q_dct[0][0] = '0' * 16
    # print(q_dct)
    
    i = 0
    j = 0
    k = 0
    while i < block_size and j < block_size:
        q_dct[i][j] = zigzag[k]
        k += 1
        
        if up == True:
            if i > 0 and j < block_size - 1:
                i -= 1
                j += 1
            elif j == block_size - 1:
                up = False
                i += 1
            else:
                up = False
                j += 1
        else:
            if i < block_size - 1 and j > 0:
                i += 1
                j -= 1
            elif i == block_size - 1:
                up = True
                j += 1
            else:
                up = True
                i += 1
        
    return q_dct
    
def run_length_encoding(array):
    encoded_data = []
    count = 1

    for i in range(1, len(array)):
        if array[i] == array[i - 1]:
            count += 1
        else:
            encoded_data.extend([count, array[i - 1]])
            count = 1

    encoded_data.extend([count, array[-1]])
    return encoded_data

def run_length_decoding(encoded_data):
    decoded_data = []
    
    count = encoded_data[0:-1:2]
    value = encoded_data[1:-1:2]
    value.append(encoded_data[-1])
    
    assert len(count) == len(value)
    for i in range(len(count)):
        decoded_data.extend([value[i]]*count[i])

    return decoded_data

def JPEG(lena):
    decompressed_image = np.zeros(lena.shape)
    length = lena.shape[0]
    for i in tqdm(range(0, length, 8)):
        for j in range(0, length, 8):
            block = lena[i:i+8, j:j+8]
            
            dct = DCT(block).astype(np.int32)
    
            quantized_block = quantization(dct)
            
            zigzag = zigzag_scan(quantized_block)
            
            rle = run_length_encoding(zigzag)
            
            rld = run_length_decoding(rle)
            
            inv_zigzag = inverse_zigzag_scan(rld, block.shape[0])
            
            dequantized_block = dequantization(inv_zigzag)
            
            decompressed_block = IDCT(dequantized_block)
            
            decompressed_image[i:i+8, j:j+8] = decompressed_block
            
    return decompressed_image

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 

"""
for each block
block -> (DCT) -> dct -> (Binary) -> bin_dct -> (zigzag_scan) -> zigzag -> (run_length_encoding) -> rle == compressed_image
decompressed_block <- (IDCT) <- dec_dct <- (Decimal) <- inv_zigzag <- (inverse_zigzag_scan) <- rld <- (run_length_decoding) <- rle
"""

luminance_quantization_table = np.array([
    [6, 4, 4, 6, 10, 16, 20, 24],
    [5, 5, 6, 8, 10, 23, 24, 22],
    [6, 5, 6, 10, 16, 23, 28, 22],
    [6, 7, 9, 12, 20, 35, 32, 25],
    [7, 9, 15, 22, 27, 44, 41, 31],
    [10, 14, 22, 26, 32, 42, 45, 37],
    [20, 26, 31, 35, 41, 48, 48, 40],
    [29, 37, 38, 39, 45, 40, 41, 40]
])

chrominance_quantization_table = np.array([
    [7, 7, 8, 17, 17, 40, 42, 42],
    [7, 8, 10, 22, 21, 45, 52, 44],
    [9, 11, 19, 26, 26, 49, 56, 47],
    [14, 13, 26, 31, 40, 58, 60, 51],
    [18, 25, 40, 41, 48, 69, 69, 56],
    [24, 40, 51, 61, 60, 70, 78, 64],
    [40, 58, 60, 55, 64, 81, 104, 84],
    [51, 60, 70, 70, 78, 95, 91, 94]
])

lena = cv2.imread("lena.png")
lena_ycbcr = cv2.cvtColor(lena, cv2.COLOR_BGR2YCrCb)
cv2.imwrite("lena_ycbcr.png", lena_ycbcr)
Y, Cr, Cb = cv2.split(lena_ycbcr)
decompressed_color_image = np.zeros(lena_ycbcr.shape).astype(np.int32)


for channel in range(lena.shape[2]):
    decompressed_ch = JPEG(lena[:, :, channel])
    decompressed_color_image[:, :, channel] = decompressed_ch
    
# plt.imshow(decompressed_color_image)
# plt.show()

cv2.imwrite("lenajpg.jpg", lena)
decompressed_color_image = decompressed_color_image.astype(np.uint8)
result = cv2.cvtColor(decompressed_color_image, cv2.COLOR_YCrCb2RGB)
cv2.imwrite("lena_rgb.jpg", decompressed_color_image)

psnr = PSNR(lena_ycbcr, decompressed_color_image)
import sys
dd = cv2.imread("lena_rgb.jpg")
compression_ratio = sys.getsizeof(dd) / sys.getsizeof(lena_ycbcr)
# from skimage.measure import compare_ssim as ssim
# ssim_val = ssim(lena.astype(float), result.astype(float))
print("psnr = ", psnr)
print("CR = ", compression_ratio)
# print("SSIM = ", ssim)