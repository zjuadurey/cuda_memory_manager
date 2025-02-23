#include"use_memory_manager.h"

// 全局内存管理器实例
NMemoryManager memoryManager;

// 自定义的cudaMalloc和cudaFree
extern "C" cudaError_t myCudaMalloc(void** devPtr, size_t size) {
    if (devPtr == nullptr) {
        return cudaErrorInvalidValue; // 如果 devPtr 为空，返回无效值错误
    }

    //const uint64_t alignment_mask = static_cast<uint64_t>(256) - 1; // 0xFF

    //size = (size + alignment_mask) & ~alignment_mask;

    void* ptr = memoryManager.allocate(size);
    if (ptr == nullptr) {
        return cudaErrorMemoryAllocation; // 如果分配失败Stream Compaction cost，返回内存分配错误
    }

    *devPtr = ptr; // 将分配的内存地址存储到 devPtr 中

    return cudaSuccess; // 返回成功
}

extern "C" cudaError_t myCudaFree(void* devPtr) {
    cudaDeviceSynchronize();
    if (devPtr == nullptr) {
        return cudaErrorInvalidValue; // 如果 devPtr 为空，返回无效值错误
    }

    memoryManager.deallocate(devPtr); // 释放内存
    return cudaSuccess; // 返回成功
}