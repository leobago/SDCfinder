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

Filename    : MemoryReliability_defs.c
Authors     : Ferad Zyulkyarov, Kai Keller, Pau Farré, Leonardo Bautista-Gomez
Version     :
Copyright   :
Description : A daemon which tests the memory for errors.

This file provides constants and static variables initialization.
*/

#include "MemoryReliability_decl.h"

/*
   +===========================================================================+
   |    DEFINITIONS                                                            |
   +===========================================================================+
   */

const unsigned int  TCC_ACT_TEMP        = 100;                      // TCC activation temperature
const unsigned int  TEMP_FIELD_LOW_BIT  = 16;                       // low bit of 6 bit temp value
const unsigned int  TEMP_FIELD_OFFSET   = 412;                      // offset in /dev/cpu/#cpu/msr file
uint64_t            TEMP_FIELD_MASK     = 0x00000000003f0000;       // selects 6 bit field[22:16]
const unsigned int  KILO                = 1024;                     // KB factor
const unsigned int  MEGA                = 1048576;                  // MB factor
const unsigned int  GIGA                = 1073741824;               // GB factor
const unsigned int  MAX_ITER            = 1000;

char                HostName[255]       = "";

unsigned long long  NumBytesCPU         = 0; 
unsigned long long  NumBytesGPU         = 0;  
void*               cpu_mem             = NULL;
void*               gpu_mem             = NULL;

unsigned char       mem_pattern         = '\0';
bool                CheckCPU            = true;
bool                CheckGPU            = false;

/*
   +===========================================================================+
   |    STATIC VARIABLES                                                       |
   +===========================================================================+
   */


char                PidFileName[255]    = "pid.txt";                // stores the pid of daemon
char                OutFile[255]        = "MemoryReliability.log";  // Memory error log file
char                WarnFile[255]       = "MemoryReliability.warn"; // Logs if error exceeds warning rate 
char                ErrFile[255]        = "MemoryReliability.err";  // Logs runtime errors
char                DbgFile[255]        = "MemoryReliability.dbg";  // Logs debug information

unsigned int        WarnRate            = 0;                        // Threshold that triggers a warning
unsigned int        NumErrors           = 0;                        // Current number of errors

void*               Mem                 = NULL;                     // Keeps the allocated memory region to check

unsigned int        SleepTime           = 0;                        // Pause between checking for errors
bool                ExitNow             = false;                    // Triggers daemon to exit 
int                 IsDaemonStart       = 0;                        // Run as daemon if 1
int                 IsDaemonStop        = 0;                        // Kill Daemon if 1
