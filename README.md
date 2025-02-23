# cuda_memory_manager
light, user-friendly memory manager for the CUDA programs.

use std::set to manage the memory fragments.

# usage

0. Before you use it, please review the file kernel.cu in this rep. It won't cost much time but benefits the usage.

1. Add the folder "cuda_memory_manager" to the root of your project.

2. In the .cu files where you wanna use this memory manager, add the line:
`
#include "cuda_memory_manager/use_memory_manager.h"
`
before #include <cuda_runtime.h> and thrust includes (if exist)

3. That's all!

# reference

This code is a reproduction of the following article:

https://www.canaknesil.com/docs/MAM_A_Memory_Allocation_Manager_for_GPUs.pdf
