#include"use_memory_manager.h"

// ȫ���ڴ������ʵ��
NMemoryManager memoryManager;

// �Զ����cudaMalloc��cudaFree
extern "C" cudaError_t myCudaMalloc(void** devPtr, size_t size) {
    if (devPtr == nullptr) {
        return cudaErrorInvalidValue; // ��� devPtr Ϊ�գ�������Чֵ����
    }

    //const uint64_t alignment_mask = static_cast<uint64_t>(256) - 1; // 0xFF

    //size = (size + alignment_mask) & ~alignment_mask;

    void* ptr = memoryManager.allocate(size);
    if (ptr == nullptr) {
        return cudaErrorMemoryAllocation; // �������ʧ��Stream Compaction cost�������ڴ�������
    }

    *devPtr = ptr; // ��������ڴ��ַ�洢�� devPtr ��

    return cudaSuccess; // ���سɹ�
}

extern "C" cudaError_t myCudaFree(void* devPtr) {
    cudaDeviceSynchronize();
    if (devPtr == nullptr) {
        return cudaErrorInvalidValue; // ��� devPtr Ϊ�գ�������Чֵ����
    }

    memoryManager.deallocate(devPtr); // �ͷ��ڴ�
    return cudaSuccess; // ���سɹ�
}