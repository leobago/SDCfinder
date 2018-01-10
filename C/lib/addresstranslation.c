/*
 * addresstranslation.c
 *
 * Implements address translation from virtual to physical.
 * (see: http://fivelinesofcode.blogspot.com.es/2014/03/how-to-translate-virtual-to-physical.html)
 *
 */

#include "addresstranslation.h"
#include "MemoryReliability_decl.h"

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <inttypes.h>

//#define DEBUG_ON
#define PAGE_SIZE sysconf(_SC_PAGESIZE)
#define PAGEMAP_ENTRY 8 // entry size in bytes
#define ENTRY_SIZE 64 // entry size in bits
#define GET_BIT(X,Y) (X & ((uint64_t)1<<Y)) >> Y
#define GET_PFN(X) X & 0x7FFFFFFFFFFFFF 
#define GET_OFFSET(X) X & ( PAGE_SIZE -1 )
#define MAX_ITER 1000

// This file contains the page table (mapping virtual address -> physical address)
#define page_mapping_file "/proc/self/pagemap"

const int __endian_bit = 1;
#define is_bigendian() ( (*(char*)&__endian_bit) == 0 )


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
        dprintf(fd_err, "Cannot open %s. Please, run as root: -- %s\n", page_mapping_file, strerror(ierr));
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
