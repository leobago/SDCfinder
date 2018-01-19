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

/** \file   logging.c
 *  \author Ferad Zyulkyarov
 *  \author Kai Keller
 *  \author Pau Farré
 *  \author Leonardo Bautista-Gomez
 *  \brief  Provides functions debug, message, even and error logging.
 */

#include "logging.h"

int read_temperature()
{
    int fd, fd_err, core = 0, ierr;
    uint64_t temp;
    char msr_file[32];
    memset(msr_file, 0, 32);

#ifdef SYS_getcpu
    if (syscall(SYS_getcpu, &core, NULL, NULL) < 0) {
        core = 0;
    }
#endif

    fd_err = open(ErrFile, O_WRONLY|O_CREAT|O_APPEND, 0666);

    snprintf(msr_file, 31, "/dev/cpu/%u/msr", core);
    fd = open(msr_file, O_RDONLY);
    if (fd == -1) {
        ierr = errno;
        dprintf(fd_err, "could not OPEN msr file: %s\n", strerror(ierr));
        ierr = 0;
        return -1;
    }

    ierr = pread(fd, &temp, sizeof(temp), TEMP_FIELD_OFFSET);
    if (ierr == -1) {
        ierr = errno;
        dprintf(fd_err, "could not READ msr file: %s\n", strerror(ierr));
        ierr = 0;
        return -1;
    }

    temp &= TEMP_FIELD_MASK;
    temp >>= TEMP_FIELD_LOW_BIT;
    temp = TCC_ACT_TEMP - temp;

    close(fd_err);
    close(fd);

    return temp;
}

//! \internal [get_formatted_timestamp]
time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes)
{
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    info = localtime(&rawtime);
    strftime(out_time_formatted, num_bytes, "%x - %H:%M:%S", info);
    return rawtime;
}
//! \internal [get_formatted_timestamp]

void warn_for_errors()
{
	FILE *f = fopen(OutFile, "a+");
	if (f)
	{
		fprintf(f, "%s\n", HostName);
		fclose(f);
	}
}

//! \internal [log_error]
void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value)
{
	char time_str[255];
	time_t time = get_formatted_timestamp(time_str, 255);
	int temperature = read_temperature();
	ADDRVALUE physical_address = 0;
	NumErrors++;

	FILE *f = fopen(OutFile, "a+");
	if (!f)
	{
        printf("ERROR, something went wrong opening file %s!: %s\n", OutFile, strerror(errno));
		exit(EXIT_FAILURE);
	}
    int field_width_value = 16;
	physical_address = virtual_to_physical_address((ADDRVALUE)address);
	fprintf(f, "%s,%lld,ERROR,%s,%p,0x%*" PRIxPTR ",0x%*" PRIxPTR ",%d,0x%" PRIxPTR "\n", 
            time_str, (long long int)time, HostName, address, field_width_value, 
            actual_value, field_width_value, expected_value, temperature, 
            physical_address);
	fclose(f);

	if (NumErrors == WarnRate)
	{
		warn_for_errors();
	}
}
//! \internal [log_error]

//! \internal [log_message]
void log_message(char* message)
{
	char time_str[255];
	time_t time = get_formatted_timestamp(time_str, 255);
	int temperature = 0;

	FILE *f = fopen(OutFile, "a+");
	if (!f)
	{
        printf("ERROR, something went wrong opening file %s!: %s\n", OutFile, strerror(errno));
		exit(EXIT_FAILURE);
	}
	temperature = read_temperature();

	fprintf(f, "%s,%lld,%s,%s,%d\n", time_str, (long long int)time, message, HostName, temperature);
	fclose(f);
}
//! \internal [log_message]
