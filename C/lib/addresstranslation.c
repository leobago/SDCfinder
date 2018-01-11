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

Filename    : addresstranslation.c
Authors     : Ferad Zyulkyarov, Kai Keller, Pau Farré, Leonardo Bautista-Gomez
Version     :
Copyright   :
Description : A daemon which tests the memory for errors.

This file provides a function to translate a virtual address to the physical
address.
*/

#include "addresstranslation.h"

uintptr_t virtual_to_physical_address(uintptr_t virt_addr) {

    uintptr_t vpfn = 0; // virtual page frame number
    uintptr_t pfn = 0; // physical page frame number
    int pfo = 0; // page frame offset
    int pfo_width = 0; // page frame offset width
    uintptr_t phys_addr = 0, phys_addr_big_endian = 0;
    uintptr_t file_offset = 0;
    uint64_t page_table_entry = 0, page_table_entry_big_endian = 0;
    int c = 1, i;
    int fd, fd_err, fd_dbg, ierr;

    if(PAGE_SIZE <= 0) {
        return 0;
    }

    fd = open(page_mapping_file, O_RDONLY, 0);
    if (fd == -1) {
        ierr = errno;
        fd_err = open(ErrFile, O_WRONLY|O_CREAT|O_APPEND, 0666);
        dprintf(fd_err, "[Warning] Cannot open %s. Please, run as root: -- %s\n", page_mapping_file, strerror(ierr));
        close(fd_err);
        ierr = 0;
        return 0;
    }

    // determine virtual address page frame
    vpfn = virt_addr / PAGE_SIZE;

    // set file pointer
    file_offset = vpfn * PAGEMAP_ENTRY;

    ierr = lseek(fd, file_offset, SEEK_SET);
    if (ierr == -1) {
        ierr = errno;
        fd_err = open(ErrFile, O_WRONLY|O_CREAT|O_APPEND, 0666);
        dprintf(fd_err, "Error! Cannot seek in %s: -- %s\n", page_mapping_file, strerror(ierr));
        close(fd_err);
        ierr = 0;
        close(fd);
        return 0;
    }

    read( fd, &page_table_entry, sizeof(uint64_t) );

#ifdef DEBUG_ON
    fd_dbg = open(DbgFile, O_WRONLY|O_CREAT|O_APPEND, 0666);
    dprintf(fd_dbg, "page table entry : 0x%" PRIxPTR "\n", page_table_entry);
    dprintf(fd_dbg, "virt_addr : 0x%" PRIxPTR "\n", virt_addr);
    dprintf(fd_dbg, "page size : %li\n", PAGE_SIZE);
    dprintf(fd_dbg, "big endian : %i\n", is_bigendian());
    for(i=0; i<ENTRY_SIZE; i++) {
        page_table_entry_big_endian += GET_BIT( page_table_entry, i );
        if( i<ENTRY_SIZE-1 ) {
            page_table_entry_big_endian <<= 1;
        }
    }
    dprintf(fd_dbg, "swapped entry : 0x%" PRIxPTR "\n", page_table_entry_big_endian);
#endif
    
    // if big endian, swap entries before apply GET_PFN macro
    page_table_entry_big_endian = 0;
    if( is_bigendian() ) {
        for(i=0; i<ENTRY_SIZE; i++) {
            page_table_entry_big_endian += GET_BIT( page_table_entry, i );
            if( i<ENTRY_SIZE-1 ) {
                page_table_entry_big_endian <<= 1;
            }
        }
        page_table_entry = page_table_entry_big_endian;
    }

    pfn = GET_PFN(page_table_entry);

#ifdef DEBUG_ON
    dprintf(fd_dbg, "pfn : 0x%" PRIxPTR "\n", pfn);
#endif

    // determine width of offset
    assert( (PAGE_SIZE != 0) && (c==1) );
    while ( c != PAGE_SIZE || pfo_width == MAX_ITER ) {
        c = c << 1;
        pfo_width++;
    }
    
    pfo = GET_OFFSET(virt_addr);
    phys_addr = (pfn << pfo_width) + pfo;

    // if big endian, swap back to normal
    phys_addr_big_endian = 0;
    if( is_bigendian() ) {
        for(i=0; i<ENTRY_SIZE; i++) {
            phys_addr_big_endian += GET_BIT( phys_addr, i );
            if( i<ENTRY_SIZE-1 ) {
                phys_addr_big_endian <<= 1;
            }
        }
        phys_addr = phys_addr_big_endian;
    }

#ifdef DEBUG_ON
    close(fd_dbg);
#endif
    close(fd);

    return phys_addr;
}
