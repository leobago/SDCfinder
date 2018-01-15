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

/** \file   addresstranslation.h
 *  \author Ferad Zyulkyarov
 *  \author Kai Keller
 *  \author Pau Farré
 *  \author Leonardo Bautista-Gomez
 *  \brief  Header file for addresstranslation.c
 */

#ifndef __ADDRESS_TRANSLATION_H
#define __ADDRESS_TRANSLATION_H

#include "MemoryReliability_decl.h"

/*! page size of virtual and physical memory */
#define PAGE_SIZE sysconf(_SC_PAGESIZE)

/*! length of each page-table entry in bytes */
#define PAGEMAP_ENTRY_BYTES 8

/*! length of each page-table entry in bits */
#define PAGEMAP_ENTRY_BITS 64

/*! get bit at pos Y in X 
 *  \param  X   [in]    uint64_t value
 *  \param  Y   [out]   position of bit to return
 *  \return Bit value at position Y in X 
 *
 *  The bit position Y is counted from the right 
 **/
#define GET_BIT(X,Y) (X & ((uint64_t)1<<Y)) >> Y

/*! \brief  extract page frame number (PFN) from page table entry X 
 *  \param  X   [in]    uint64_t page-table entry
 *  \return PFN 
 *  
 *  This macro sets bits 53 - 64 of the page-table entry to zero in order
 *  to acquire the PFN.
 *
 *  notice: mask for little endian. big-endian value must be swapped first 
 */
#define GET_PFN(X) X & 0x7FFFFFFFFFFFFF 

/*! \brief  get address offset
 *  \param  X   [in]    uintptr_t virtual-address value
 *  \return offset of virtual-address value
 *
 *  This macro returns the offset to the page frame of the virtual-address value
 * */
#define GET_OFFSET(X) X & ( PAGE_SIZE -1 )

/*! security measure to prevent infinite loop execution */
#define MAX_ITER 1000

/*! file containing the page table (mapping virtual address -> physical address) */
#define page_mapping_file "/proc/self/pagemap"

const int __endian_bit = 1;
/*! check endianess */
#define is_bigendian() ( (*(char*)&__endian_bit) == 0 )


#endif

