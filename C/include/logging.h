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

/** \file   logging.h
 *  \author Ferad Zyulkyarov
 *  \author Kai Keller
 *  \author Pau Farré
 *  \author Leonardo Bautista-Gomez
 *  \brief  header file for logging.c
 */

#ifndef LOGGING_H
#define LOGGING_H

#include "MemoryReliability_decl.h"

extern const unsigned int TCC_ACT_TEMP;             /**< TCC activation temperature (=100)                  */
extern const unsigned int TEMP_FIELD_LOW_BIT;       /**< low bit of 6 bit temp value (=16)                  */
extern const unsigned int TEMP_FIELD_OFFSET;        /**< offset in /dev/cpu/#cpu/msr file (=412)            */
extern uint64_t TEMP_FIELD_MASK;                    /**< selects 6 bit field[22:16] (=0x00000000003f0000    */

/*! \brief Warns if amount of errors exceed a certain value TODO */
void warn_for_errors();

#endif //LOGGING_H
