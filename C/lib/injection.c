/*
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, Leo
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ============================================================================
*/

#include "injection.h"

void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s)
{
#ifdef INJECT_ERR
    if (s != SIMPLE)
    {
        uint64_t MAX_EXP_VAL = 0;
        unsigned int MAX_EXP = 0;

        while( ( (MAX_EXP_VAL<<1) < NumBytesCPU ) && ( MAX_EXP < MAX_ITER ) ) {
            MAX_EXP_VAL = (uint64_t) 1 << MAX_EXP;
            MAX_EXP++;
        }
 
        int riter = rand()%10 + 1;
        if (riter == 5) {
            ru64 r_gen;
            ADDRVALUE r_off;
            r_gen.p1 = rand();
            r_gen.p2 = rand();
            memcpy(&r_off, &r_gen, sizeof(ADDRVALUE));
            r_off &= r_off_masks[rand()%MAX_EXP];
            r_off = (r_off%NumBytesCPU);
            unsigned char *ptr_data = start + r_off;
            if (s == ZERO) {
                memset(ptr_data, 0x1, 1);
            }
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask);
            if (s == RANDOM) {
                *ptr_data = ~expected_value;
            }
            char msg[1000];
            snprintf(msg,1000,"Inject error at: %p (CPU), memory allocation starts at: %p", ptr_data, start);
            log_message(msg);
        }
    }
#endif
}

void inject_gpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s)
{
#if defined(INJECT_ERR) && defined(USE_CUDA)
    if (s != SIMPLE)
    {
        int riter = rand()%10 + 1;
        if (riter == 5) {
            unsigned int ruibytes;
            int ribytes = rand();
            memcpy(&ruibytes, &ribytes, sizeof(ribytes));
            ruibytes = (ruibytes%NumBytesGPU);
            unsigned char* ptr_data = start + ruibytes;
            if (s == ZERO) {
                cudaError_t err = cudaMemset(ptr_data, 0x1, 1);
                if (err != cudaSuccess)
                {
                    char msg[255];
                    sprintf(msg, "ERROR,Cannot memset ptr_data to 0x1 (%s:%s).",
                            cudaGetErrorName(err), cudaGetErrorString(err));
                    log_message(msg);
                }
            }
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask);
            ADDRVALUE new_value = ~expected_value;
            if (s == RANDOM) {
                cudaError_t err = cudaMemcpy(ptr_data, &new_value, sizeof(ADDRVALUE), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    char msg[255];
                    sprintf(msg, "ERROR,Cannot memcpy ptr_data to GPU (%s:%s).",
                            cudaGetErrorName(err), cudaGetErrorString(err));
                    log_message(msg);
                }
            }
            char msg[1000];
            snprintf(msg,1000,"Inject error at %p (GPU)", ptr_data);
            log_message(msg);
        }
    }
#endif
}

