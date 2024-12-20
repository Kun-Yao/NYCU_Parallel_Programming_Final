# Accelerate JPEG Compression with Parallel Programming

## 使用方法

### 選擇平行化方法
cd 至欲平行化的方法資料夾(CUDA/Pthread/OpenMP)

### 編譯

```
make
```

### 執行

* CUDA version
```
srun ./JPEGCompress_main -f <input_image_path> -o <output_image_path>
```
input_image_path 不可為空，需加上副檔名  
output_image_path預設為output.png，可以自定義，需加上副檔名

* Pthread version
```
srun -c 6 ./JPEGCompress_main <num_threads> -f <input_image_path> -o <output_image_path>
```
num_threads必須輸入正整數，不能空白  
input_image_path 不可為空，需加上副檔名  
output_image_path預設為output.png，可以自定義，需加上副檔名  

* OpenMP version
```
srun -c <num_threads> ./JPEGCompress_main -f <input_image_path> -o <output_image_path>
```
num_threads必須輸入正整數，不能空白  
input_image_path 不可為空，需加上副檔名  
output_image_path預設為output.png，可以自定義，需加上副檔名  

## 輸入圖片限制
圖片部分目前僅支援 bmp 格式，且必須為 24 bit 的 rgb bmp 檔案
