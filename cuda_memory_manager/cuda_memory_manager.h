//16
#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <iostream>
#include <set> // ʹ�� std::set ��� RBTree

// ���� Segment �ṹ��
struct Segment {
    void* base;
    size_t size;
    bool isEmpty;

    // ȷ��ָ������ʹ�� char*
    char* baseAsChar() const {
        return static_cast<char*>(base);
    }

    // ���� < ����������� std::set ������
    bool operator<(const Segment& other) const {
        return base < other.base;
    }
};

// �Ƚ��ࣺ���ڴ�С
struct CompareSize {
    bool operator()(const Segment& a, const Segment& b) const {
        // �ȱȽϴ�С����С��ͬ��Ƚϻ���ַ
        if (a.size == b.size)
            return a.base < b.base;  // ȷ����ͬ��С�Ĳ�ͬ�β�����Ϊ�ȼ�
        return a.size < b.size;
    }
};

class NMemoryManager {
public:
    NMemoryManager();
    ~NMemoryManager();

    void* allocate(size_t size);
    void deallocate(void* ptr);

    void createBlock(); // ��ʼ���ڴ��
    void freeBlock();   // �ͷ��ڴ��
    void printMemoryPool(); // ��ӡ�ڴ��

private:
    void* chunk; // ȫ���ڴ��
    size_t chunkSize; // �ڴ�ش�С

    std::set<Segment> ptrTree;    // ���ڵ�ַ����
    std::set<Segment, CompareSize> sizeTreeDic; // ���ڴ�С����

    Segment* findBestFit(size_t size);
    void splitSegment(Segment* seg, size_t size);
    void mergeSegments(const Segment& segToMerge);

    void printTreeInOrder(const std::set<Segment>& tree, int& count) const;
};

#endif // N_MEMORY_MANAGER_H
