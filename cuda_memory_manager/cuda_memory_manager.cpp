//16
#include "cuda_memory_manager.h"

// 构造函数
NMemoryManager::NMemoryManager() : chunk(nullptr), chunkSize(0) {
    printf("NMemoryManager()\n");
}

// 析构函数
NMemoryManager::~NMemoryManager() {
    printf("~NMemoryManager()\n");  
    freeBlock(); // 释放内存池
}

// 初始化内存池
void NMemoryManager::createBlock() {
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    chunkSize = free - (512 * 1024 * 1024);
    //chunkSize = free; // 设置内存池大小为可用内存大小

    // 分配内存池
    err = cudaMalloc(&chunk, chunkSize);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 初始化内存段并插入到树中
    Segment seg{ chunk, chunkSize, true };
    ptrTree.insert(seg);
    sizeTreeDic.insert(seg);

    // 打印调试信息
    std::cout << "Initial segment created: base = " << seg.base << ", size = " << seg.size << std::endl;
    std::cout << "ptrTree size: " << ptrTree.size() << std::endl;
    std::cout << "sizeTreeDic size: " << sizeTreeDic.size() << std::endl;
}

// 释放内存池
void NMemoryManager::freeBlock() {
    //printMemoryPool();
    if (chunk != nullptr) {
        cudaError_t err = cudaFree(chunk);
        if (err == cudaSuccess) {
            printf("chunk cudaFree success\n");
        }
        else {
            printf("chunk cudaFree failed\n");
        }
        chunk = nullptr;
    }

    // 清空树
    ptrTree.clear();
    sizeTreeDic.clear();
}

// 打印内存池
void NMemoryManager::printMemoryPool() {
    std::cout << std::endl << "Memory Pool Status:" << std::endl;

    int segmentCount = 0;
    printTreeInOrder(ptrTree, segmentCount); // 打印基于地址的树

    std::cout << "Total Segments: " << segmentCount << std::endl << std::endl;
}

// 打印树的内容
void NMemoryManager::printTreeInOrder(const std::set<Segment>& tree, int& count) const {
    for (const auto& seg : tree) {
        count++;
        std::cout << "Segment " << count << ": "
            << "Base Address = " << seg.base << ", "
            << "Size = " << seg.size << " bytes, "
            << "Status = " << (seg.isEmpty ? "Free" : "Allocated")
            << std::endl;
    }
}

// 分配内存
void* NMemoryManager::allocate(size_t size) {
    const uint64_t alignment_mask = static_cast<uint64_t>(256) - 1; // 0xF

    size = (size + alignment_mask) & ~alignment_mask;

    Segment* seg = findBestFit(size);
    if (!seg || seg->size < size) {
        std::cerr << "No available memory block of size " << size << std::endl;
        return nullptr;
    }

    // 检查是否需要分割
    if (seg->size > size) {
        void* segBase = seg->base;
        splitSegment(seg, size); // 分割段

        //printMemoryPool();

        return segBase;
    }


    // 不分割
    Segment originalSeg = *seg;
    originalSeg.isEmpty = false;
    // 标记为已分配
    // seg->isEmpty = false;
    void* segBase = seg->base;
    ptrTree.erase(*seg);
    sizeTreeDic.erase(*seg);
    ptrTree.insert(originalSeg);
    //sizeTreeDic.insert(originalSeg);

    //printMemoryPool();

    return segBase;
}


// 释放内存
void NMemoryManager::deallocate(void* ptr) {
    if (!ptr) {
        std::cerr << "Attempt to deallocate a null pointer" << std::endl;
        return;
    }

    // 查找对应的段
    Segment key{ ptr, 0, false };
    auto it = ptrTree.find(key);
    if (it == ptrTree.end()) {
        std::cerr << "Attempt to deallocate an unknown pointer" << std::endl;
        return;
    }

    // 标记为空闲
    Segment seg = *it;
    seg.isEmpty = true;

    // 删除旧段
    ptrTree.erase(it);
    //sizeTreeDic.erase(it); //bug

    // 插入修改后的段
    ptrTree.insert(seg);
    // sizeTreeDic.insert(seg);

    // 合并相邻的空闲段
    mergeSegments(seg);
}


// 查找最佳匹配的段
Segment* NMemoryManager::findBestFit(size_t size) {
    Segment key{ nullptr, size, false };
    auto it = sizeTreeDic.lower_bound(key);
    if (it != sizeTreeDic.end()) {
        return const_cast<Segment*>(&(*it));
    }
    return nullptr;
}

// 分割段
void NMemoryManager::splitSegment(Segment* seg, size_t size) {
    // 0. 检查分割大小是否合法
    if (seg->size <= size) {
        throw std::invalid_argument("Cannot split: segment size <= requested size");
    }
    //std::cout << "seg->base= " << seg->base << ", seg->size=" << seg->size << ", size= " << size << std::endl;
    // 1. 从容器中删除旧段
    char* oldBase = (char*)(seg->base);
    size_t oldSize = seg->size;
    ptrTree.erase(*seg);
    sizeTreeDic.erase(*seg);

    // 2. 计算新段的地址和大小
    char* originalBase = oldBase;
    void* newBase = originalBase + size; // 正确的指针步进
    size_t newSize = oldSize - size;
    // std::cout << "splitSegment: " << seg->size << " " << size << " " << newSize << std::endl;
    // 3. 创建新段
    Segment modifySeg{ originalBase, size, false };
    Segment newSeg{ newBase, newSize, true };

    // 4. 更新旧段的大小
    seg->size = size;

    // 5. 插入修改后的段和新段
    ptrTree.insert(modifySeg);
    ptrTree.insert(newSeg);
    //sizeTreeDic.insert(modifySeg);
    sizeTreeDic.insert(newSeg);

    // 调试信息
    // std::cout << "Split segment: base=" << seg->base << ", new_size=" << seg->size << std::endl;
    // std::cout << "New segment: base=" << newBase << ", size=" << newSize << std::endl;
}



// 合并相邻的空闲段
void NMemoryManager::mergeSegments(const Segment& segToMerge) {
    if (!segToMerge.isEmpty) return;

    // 获取当前段在两个树中的准确位置
    auto currentPtrIt = ptrTree.find(segToMerge);
    if (currentPtrIt == ptrTree.end()) return;

    // 同步获取 sizeTreeDic 中的对应项
    auto currentSizeIt = sizeTreeDic.find(*currentPtrIt);

    Segment mergedSeg = *currentPtrIt;
    bool merged = false;

    /****************** 前向合并 ******************/
    if (currentPtrIt != ptrTree.begin()) {
        auto prevPtrIt = std::prev(currentPtrIt);
        if (prevPtrIt->isEmpty) {
            // 计算地址连续性
            char* prevEnd = prevPtrIt->baseAsChar() + prevPtrIt->size;
            if (prevEnd == mergedSeg.base) {
                // 获取 sizeTreeDic 中的对应项
                auto prevSizeIt = sizeTreeDic.find(*prevPtrIt);

                // 合并段属性
                mergedSeg.base = prevPtrIt->base;
                mergedSeg.size += prevPtrIt->size;
                merged = true;

                // 双树同步删除
                ptrTree.erase(prevPtrIt);
                if (prevSizeIt != sizeTreeDic.end())
                    sizeTreeDic.erase(prevSizeIt);
            }
        }
    }

    /****************** 后向合并 ******************/
    auto nextPtrIt = std::next(currentPtrIt);
    if (nextPtrIt != ptrTree.end() && nextPtrIt->isEmpty) {
        char* currentEnd = mergedSeg.baseAsChar() + mergedSeg.size;
        if (currentEnd == nextPtrIt->base) {
            // 获取 sizeTreeDic 中的对应项
            auto nextSizeIt = sizeTreeDic.find(*nextPtrIt);

            //use nextPtrIt to update mergeSeg
            mergedSeg.size += nextPtrIt->size;
            merged = true;

            // 双树同步删除
            ptrTree.erase(nextPtrIt);
            if (nextSizeIt != sizeTreeDic.end())
                sizeTreeDic.erase(nextSizeIt);

        }
    }

    /****************** 更新双树 ******************/
    if (merged) {
        // 删除原当前段
        ptrTree.erase(currentPtrIt);
        if (currentSizeIt != sizeTreeDic.end())
            sizeTreeDic.erase(currentSizeIt);

        // 插入新合并段
        ptrTree.insert(mergedSeg);
        sizeTreeDic.insert(mergedSeg);
    }
}


