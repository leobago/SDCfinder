#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef uintptr_t ADDRVALUE;
typedef void* ADDRESS;

typedef enum Strategy {
    SIMPLE = 0,
    ZERO = 1,
    RANDOM = 2
} Strategy_t;

#endif //TYPES_H
