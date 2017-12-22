#include "logging.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>

#include <inttypes.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "addresstranslation.h"

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

time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes)
{
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    info = localtime(&rawtime);
    strftime(out_time_formatted, num_bytes, "%x - %H:%M:%S", info);
    return rawtime;
}

void warn_for_errors()
{
	FILE *f = fopen(OutFile, "a+");
	if (f)
	{
		fprintf(f, "%s\n", HostName);
		fclose(f);
	}
}

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
	fprintf(f, "%s,%lld,ERROR,%s,%p,%*" PRIxPTR ",%*" PRIxPTR ",%d,%" PRIxPTR "\n", 
            time_str, (long long int)time, HostName, address, field_width_value, 
            actual_value, field_width_value, expected_value, temperature, 
            physical_address);
	fclose(f);

	if (NumErrors == WarningRate)
	{
		warn_for_errors();
	}
}

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

void log_local_mem_errors(int local_mem_errors)
{
	if (local_mem_errors != 0)
	{
		if (local_mem_errors < 0)
		{
			char err_msg[255] = "ERROR_INFO,Cannot read MemLocalErrs counter.";
			log_message(err_msg);
		}

		if (local_mem_errors > 0)
		{
			char err_msg[255];
			memset(err_msg, 0, 255);
			sprintf(err_msg, "ERROR_MEM_LOCAL,%d", local_mem_errors);
			log_message(err_msg);
		}
	}
}

