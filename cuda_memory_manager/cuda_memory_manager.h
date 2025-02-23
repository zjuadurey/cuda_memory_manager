//16
#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <iostream>
#include <set> // 使用 std::set 替代 RBTree

// 定义 Segment 结构体
struct Segment {
    void* base;
    size_t size;
    bool isEmpty;

    // 确保指针运算使用 char*
    char* baseAsChar() const {
        return static_cast<char*>(base);
    }

    // 重载 < 运算符，用于 std::set 的排序
    bool operator<(const Segment& other) const {
        return base < other.base;
    }
};

// 比较类：基于大小
struct CompareSize {
    bool operator()(const Segment& a, const Segment& b) const {
        // 先比较大小，大小相同则比较基地址
        if (a.size == b.size)
            return a.base < b.base;  // 确保相同大小的不同段不被视为等价
        return a.size < b.size;
    }
};

class NMemoryManager {
public:
    NMemoryManager();
    ~NMemoryManager();

    void* allocate(size_t size);
    void deallocate(void* ptr);

    void createBlock(); // 初始化内存池
    void freeBlock();   // 释放内存池
    void printMemoryPool(); // 打印内存池

private:
    void* chunk; // 全局内存池
    size_t chunkSize; // 内存池大小

    std::set<Segment> ptrTree;    // 基于地址的树
    std::set<Segment, CompareSize> sizeTreeDic; // 基于大小的树

    Segment* findBestFit(size_t size);
    void splitSegment(Segment* seg, size_t size);
    void mergeSegments(const Segment& segToMerge);

    void printTreeInOrder(const std::set<Segment>& tree, int& count) const;
};

#endif // N_MEMORY_MANAGER_H
