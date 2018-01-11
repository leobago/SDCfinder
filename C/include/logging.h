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

Filename    : logging.h
Authors     : Ferad Zyulkyarov, Kai Keller, Pau Farré, Leonardo Bautista-Gomez
Version     :
Copyright   :
Description : A daemon which tests the memory for errors.

Header file for logging.h
*/

#ifndef LOGGING_H
#define LOGGING_H

#include "MemoryReliability_decl.h"

extern const unsigned int TCC_ACT_TEMP;//         = 100; // TCC activation temperature
extern const unsigned int TEMP_FIELD_LOW_BIT;//   = 16; // low bit of 6 bit temp value
extern const unsigned int TEMP_FIELD_OFFSET;//    = 412; // offset in /dev/cpu/#cpu/msr file
extern uint64_t TEMP_FIELD_MASK;//                = 0x00000000003f0000; // selects 6 bit field[22:16]

void warn_for_errors();

#endif //LOGGING_H
