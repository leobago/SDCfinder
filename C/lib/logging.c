#include "MemoryReliability_decl.h"

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

