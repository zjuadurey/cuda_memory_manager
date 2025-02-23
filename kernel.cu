// use this include to enable the memory manager to override the cudaMalloc & cudaFree
#include "cuda_memory_manager/use_memory_manager.h"

#include <iostream>
#include <string>
#include"loadTiff.h"
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>






int main() {
    // 初始化内存池
    memoryManager.createBlock();
    memoryManager.printMemoryPool();

    //读图


    std::string inputName = "test_2_uint8.tif";
    //std::string inputName = "fix-P7-4.5h-cell2-60x-zoom1.5_merge_c2.tif";
    

    int* imageShape = new int[3];
    unsigned char* h_imagePtr = loadImage(inputName, imageShape);
    int width = imageShape[0]; //963
    int height = imageShape[1]; //305
    int slice = imageShape[2]; //140


    unsigned char* d_imagePtr;
    cudaMalloc((void**)&d_imagePtr, sizeof(unsigned char) * width * height * slice);

    cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyHostToDevice);

    //double sum = thrust::reduce(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, 0.0, thrust::plus<double>());
    //sum = sum / (width * height * slice);

    //printf("sum: %lf", sum);

    memoryManager.printMemoryPool();

    cudaFree(d_imagePtr);

    memoryManager.printMemoryPool();

    // 分配内存
    int* d_data_1;
    size_t size = 3818206720;
    cudaError_t err = cudaMalloc((void**)&d_data_1, size);
    if (err != cudaSuccess) {
        std::cerr << "myCudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    memoryManager.printMemoryPool();


    int* d_data_2;
    size = 2 * 1024;
    err = myCudaMalloc((void**)&d_data_2, size);
    if (err != cudaSuccess) {
        std::cerr << "myCudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    memoryManager.printMemoryPool();

    int* d_data_3;
    size = 1024 * 1024 * 1024;
    err = myCudaMalloc((void**)&d_data_3, size);

    memoryManager.printMemoryPool();

    // 释放内存
    err = myCudaFree(d_data_2);
    if (err != cudaSuccess) {
        std::cerr << "myCudaFree failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    memoryManager.printMemoryPool();

    err = myCudaFree(d_data_1);
    if (err != cudaSuccess) {
        std::cerr << "myCudaFree failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    memoryManager.printMemoryPool();

    err = myCudaFree(d_data_3);
    if (err != cudaSuccess) {
        std::cerr << "myCudaFree failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    memoryManager.printMemoryPool();

    // 释放内存池
    memoryManager.freeBlock();

    std::cout << "Memory management test completed successfully!" << std::endl;
    return 0;
}
