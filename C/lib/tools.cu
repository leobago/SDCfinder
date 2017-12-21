#include <cstdint>
#include <cstdio>

#include "MemoryReliability_decl.cuh"

__global__
void check_gpu_kernel(uintptr_t* buffer,
                      const long unsigned num_words,
                      uintptr_t expected_value,
                      uintptr_t new_value)
{
    for (long unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
                       word < num_words;
                       word += gridDim.x * blockDim.x)
    {
        uintptr_t actual_value = buffer[word];
        if (actual_value != expected_value)
        {
            printf("ERROR, %p, %p, %p\n", &buffer[word], actual_value, expected_value);
            //log_error(&buffer[word], actual_value, expected_value);
        }

        buffer[word] = new_value;
    }
}

void check_gpu_stub(uintptr_t* buffer,
                    unsigned long long num_bytes,
                    uintptr_t expected_value,
                    uintptr_t new_value)
{
    const long unsigned num_words = num_bytes / sizeof(uintptr_t);
    const int block_size = 128;
    const int grid_size = (num_words + block_size - 1) / block_size;
    check_gpu_kernel<<<grid_size, block_size>>>
            (buffer, num_words, expected_value, new_value);
}