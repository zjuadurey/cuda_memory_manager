// use this include to enable the memory manager to override the cudaMalloc & cudaFree
#include "cuda_memory_manager/use_memory_manager.h"

#include <iostream>
#include <string>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>


int main() {
    // 初始化内存池
    memoryManager.createBlock();

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
