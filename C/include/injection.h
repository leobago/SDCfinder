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

#ifndef _INJECTION_H
#define _INJECTION_H

#include "MemoryReliability_decl.h"

typedef struct _ru64 {
    uint32_t p1;
    uint32_t p2;
} ru64;

uint64_t r_off_masks[64] = {
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
    0x3FFFFFFFFFFFFFFF,
    0x1FFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFF,
    0x3FFFFFFFFFFFFFF,
    0x1FFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFF,
    0x3FFFFFFFFFFFFF,
    0x1FFFFFFFFFFFFF,
    0xFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFF,
    0x3FFFFFFFFFFFF,
    0x1FFFFFFFFFFFF,
    0xFFFFFFFFFFFF,
    0x7FFFFFFFFFFF,
    0x3FFFFFFFFFFF,
    0x1FFFFFFFFFFF,
    0xFFFFFFFFFFF,
    0x7FFFFFFFFFF,
    0x3FFFFFFFFFF,
    0x1FFFFFFFFFF,
    0xFFFFFFFFFF,
    0x7FFFFFFFFF,
    0x3FFFFFFFFF,
    0x1FFFFFFFFF,
    0xFFFFFFFFF,
    0x7FFFFFFFF,
    0x3FFFFFFFF,
    0x1FFFFFFFF,
    0xFFFFFFFF,
    0x7FFFFFFF,
    0x3FFFFFFF,
    0x1FFFFFFF,
    0xFFFFFFF,
    0x7FFFFFF,
    0x3FFFFFF,
    0x1FFFFFF,
    0xFFFFFF,
    0x7FFFFF,
    0x3FFFFF,
    0x1FFFFF,
    0xFFFFF,
    0x7FFFF,
    0x3FFFF,
    0x1FFFF,
    0xFFFF,
    0x7FFF,
    0x3FFF,
    0x1FFF,
    0xFFF,
    0x7FF,
    0x3FF,
    0x1FF,
    0xFF,
    0x7F,
    0x3F,
    0x1F,
    0xF,
    0x7,
    0x3,
    0x1
};

#endif

