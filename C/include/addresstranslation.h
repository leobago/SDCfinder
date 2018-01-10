/*
 * addresstranslation.h
 *
 *  Translates virtual to physical address.
 */

#ifndef __ADDRESS_TRANSLATION_H
#define __ADDRESS_TRANSLATION_H

#include <stdint.h>
#include <sys/mman.h>
#include <assert.h>

uintptr_t virtual_to_physical_address(uintptr_t virt_addr);

#endif

