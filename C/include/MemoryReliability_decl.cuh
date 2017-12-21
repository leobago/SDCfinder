#ifndef _MEMORYRELIABILITY_DECL_CUH
#define _MEMORYRELIABILITY_DECL_CUH

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void check_gpu_stub(uintptr_t* buffer,
                    unsigned long long num_bytes,
                    uintptr_t expected_value,
                    uintptr_t new_value);

#ifdef __cplusplus
};
#endif

#endif