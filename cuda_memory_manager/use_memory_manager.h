#pragma once
// use these two lines before  #include <cuda_runtime.h> to override the cudaMalloc & cudaFree
#define cudaMalloc myCudaMalloc
#define cudaFree myCudaFree

#include <vector>
#include <iostream>
#include "cuda_memory_manager.h"

// ȫ���ڴ������ʵ��
extern NMemoryManager memoryManager;

// �Զ����cudaMalloc��cudaFree
extern "C" cudaError_t myCudaMalloc(void** devPtr, size_t size);

extern "C" cudaError_t myCudaFree(void* devPtr);