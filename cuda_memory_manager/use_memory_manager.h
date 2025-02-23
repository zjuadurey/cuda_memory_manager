#pragma once
// use these two lines before  #include <cuda_runtime.h> to override the cudaMalloc & cudaFree
#define cudaMalloc myCudaMalloc
#define cudaFree myCudaFree

#include <vector>
#include <iostream>
#include "cuda_memory_manager.h"

// 全局内存管理器实例
extern NMemoryManager memoryManager;

// 自定义的cudaMalloc和cudaFree
extern "C" cudaError_t myCudaMalloc(void** devPtr, size_t size);

extern "C" cudaError_t myCudaFree(void* devPtr);