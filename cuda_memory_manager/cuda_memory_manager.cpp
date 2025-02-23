//16
#include "cuda_memory_manager.h"

// ���캯��
NMemoryManager::NMemoryManager() : chunk(nullptr), chunkSize(0) {
    printf("NMemoryManager()\n");
}

// ��������
NMemoryManager::~NMemoryManager() {
    printf("~NMemoryManager()\n");  
    freeBlock(); // �ͷ��ڴ��
}

// ��ʼ���ڴ��
void NMemoryManager::createBlock() {
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    chunkSize = free - (512 * 1024 * 1024);
    //chunkSize = free; // �����ڴ�ش�СΪ�����ڴ��С

    // �����ڴ��
    err = cudaMalloc(&chunk, chunkSize);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // ��ʼ���ڴ�β����뵽����
    Segment seg{ chunk, chunkSize, true };
    ptrTree.insert(seg);
    sizeTreeDic.insert(seg);

    // ��ӡ������Ϣ
    std::cout << "Initial segment created: base = " << seg.base << ", size = " << seg.size << std::endl;
    std::cout << "ptrTree size: " << ptrTree.size() << std::endl;
    std::cout << "sizeTreeDic size: " << sizeTreeDic.size() << std::endl;
}

// �ͷ��ڴ��
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

    // �����
    ptrTree.clear();
    sizeTreeDic.clear();
}

// ��ӡ�ڴ��
void NMemoryManager::printMemoryPool() {
    std::cout << std::endl << "Memory Pool Status:" << std::endl;

    int segmentCount = 0;
    printTreeInOrder(ptrTree, segmentCount); // ��ӡ���ڵ�ַ����

    std::cout << "Total Segments: " << segmentCount << std::endl << std::endl;
}

// ��ӡ��������
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

// �����ڴ�
void* NMemoryManager::allocate(size_t size) {
    const uint64_t alignment_mask = static_cast<uint64_t>(256) - 1; // 0xF

    size = (size + alignment_mask) & ~alignment_mask;

    Segment* seg = findBestFit(size);
    if (!seg || seg->size < size) {
        std::cerr << "No available memory block of size " << size << std::endl;
        return nullptr;
    }

    // ����Ƿ���Ҫ�ָ�
    if (seg->size > size) {
        void* segBase = seg->base;
        splitSegment(seg, size); // �ָ��

        //printMemoryPool();

        return segBase;
    }


    // ���ָ�
    Segment originalSeg = *seg;
    originalSeg.isEmpty = false;
    // ���Ϊ�ѷ���
    // seg->isEmpty = false;
    void* segBase = seg->base;
    ptrTree.erase(*seg);
    sizeTreeDic.erase(*seg);
    ptrTree.insert(originalSeg);
    //sizeTreeDic.insert(originalSeg);

    //printMemoryPool();

    return segBase;
}


// �ͷ��ڴ�
void NMemoryManager::deallocate(void* ptr) {
    if (!ptr) {
        std::cerr << "Attempt to deallocate a null pointer" << std::endl;
        return;
    }

    // ���Ҷ�Ӧ�Ķ�
    Segment key{ ptr, 0, false };
    auto it = ptrTree.find(key);
    if (it == ptrTree.end()) {
        std::cerr << "Attempt to deallocate an unknown pointer" << std::endl;
        return;
    }

    // ���Ϊ����
    Segment seg = *it;
    seg.isEmpty = true;

    // ɾ���ɶ�
    ptrTree.erase(it);
    //sizeTreeDic.erase(it); //bug

    // �����޸ĺ�Ķ�
    ptrTree.insert(seg);
    // sizeTreeDic.insert(seg);

    // �ϲ����ڵĿ��ж�
    mergeSegments(seg);
}


// �������ƥ��Ķ�
Segment* NMemoryManager::findBestFit(size_t size) {
    Segment key{ nullptr, size, false };
    auto it = sizeTreeDic.lower_bound(key);
    if (it != sizeTreeDic.end()) {
        return const_cast<Segment*>(&(*it));
    }
    return nullptr;
}

// �ָ��
void NMemoryManager::splitSegment(Segment* seg, size_t size) {
    // 0. ���ָ��С�Ƿ�Ϸ�
    if (seg->size <= size) {
        throw std::invalid_argument("Cannot split: segment size <= requested size");
    }
    //std::cout << "seg->base= " << seg->base << ", seg->size=" << seg->size << ", size= " << size << std::endl;
    // 1. ��������ɾ���ɶ�
    char* oldBase = (char*)(seg->base);
    size_t oldSize = seg->size;
    ptrTree.erase(*seg);
    sizeTreeDic.erase(*seg);

    // 2. �����¶εĵ�ַ�ʹ�С
    char* originalBase = oldBase;
    void* newBase = originalBase + size; // ��ȷ��ָ�벽��
    size_t newSize = oldSize - size;
    // std::cout << "splitSegment: " << seg->size << " " << size << " " << newSize << std::endl;
    // 3. �����¶�
    Segment modifySeg{ originalBase, size, false };
    Segment newSeg{ newBase, newSize, true };

    // 4. ���¾ɶεĴ�С
    seg->size = size;

    // 5. �����޸ĺ�Ķκ��¶�
    ptrTree.insert(modifySeg);
    ptrTree.insert(newSeg);
    //sizeTreeDic.insert(modifySeg);
    sizeTreeDic.insert(newSeg);

    // ������Ϣ
    // std::cout << "Split segment: base=" << seg->base << ", new_size=" << seg->size << std::endl;
    // std::cout << "New segment: base=" << newBase << ", size=" << newSize << std::endl;
}



// �ϲ����ڵĿ��ж�
void NMemoryManager::mergeSegments(const Segment& segToMerge) {
    if (!segToMerge.isEmpty) return;

    // ��ȡ��ǰ�����������е�׼ȷλ��
    auto currentPtrIt = ptrTree.find(segToMerge);
    if (currentPtrIt == ptrTree.end()) return;

    // ͬ����ȡ sizeTreeDic �еĶ�Ӧ��
    auto currentSizeIt = sizeTreeDic.find(*currentPtrIt);

    Segment mergedSeg = *currentPtrIt;
    bool merged = false;

    /****************** ǰ��ϲ� ******************/
    if (currentPtrIt != ptrTree.begin()) {
        auto prevPtrIt = std::prev(currentPtrIt);
        if (prevPtrIt->isEmpty) {
            // �����ַ������
            char* prevEnd = prevPtrIt->baseAsChar() + prevPtrIt->size;
            if (prevEnd == mergedSeg.base) {
                // ��ȡ sizeTreeDic �еĶ�Ӧ��
                auto prevSizeIt = sizeTreeDic.find(*prevPtrIt);

                // �ϲ�������
                mergedSeg.base = prevPtrIt->base;
                mergedSeg.size += prevPtrIt->size;
                merged = true;

                // ˫��ͬ��ɾ��
                ptrTree.erase(prevPtrIt);
                if (prevSizeIt != sizeTreeDic.end())
                    sizeTreeDic.erase(prevSizeIt);
            }
        }
    }

    /****************** ����ϲ� ******************/
    auto nextPtrIt = std::next(currentPtrIt);
    if (nextPtrIt != ptrTree.end() && nextPtrIt->isEmpty) {
        char* currentEnd = mergedSeg.baseAsChar() + mergedSeg.size;
        if (currentEnd == nextPtrIt->base) {
            // ��ȡ sizeTreeDic �еĶ�Ӧ��
            auto nextSizeIt = sizeTreeDic.find(*nextPtrIt);

            //use nextPtrIt to update mergeSeg
            mergedSeg.size += nextPtrIt->size;
            merged = true;

            // ˫��ͬ��ɾ��
            ptrTree.erase(nextPtrIt);
            if (nextSizeIt != sizeTreeDic.end())
                sizeTreeDic.erase(nextSizeIt);

        }
    }

    /****************** ����˫�� ******************/
    if (merged) {
        // ɾ��ԭ��ǰ��
        ptrTree.erase(currentPtrIt);
        if (currentSizeIt != sizeTreeDic.end())
            sizeTreeDic.erase(currentSizeIt);

        // �����ºϲ���
        ptrTree.insert(mergedSeg);
        sizeTreeDic.insert(mergedSeg);
    }
}


