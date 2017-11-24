#include "../include/MemoryReliability_decl.h"

void simple_memory_test(void* mem, unsigned int num_bytes)
{
	unsigned int i = 1;
	unsigned int word_ind = 0;

	ADDRVALUE expected_value;
	ADDRVALUE* actual_value;
	unsigned char int_size = sizeof(ADDRVALUE);
	unsigned int num_words = num_bytes / int_size;
	ADDRVALUE* buffer = NULL;

	//
	// Start the HW counters for the LocalMemErrors
	//
	int local_mem_errors = startLocalMemErrorsCounters();
	if (!local_mem_errors)
	{
		char msg[] = "Cannot start LocalMemError counters.";
		log_message(msg);
		//ExitNow = 1;
		if (daemon_pid_file_exists())
		{
			daemon_pid_delete_file();
		}
	}

	//
	// The main loop
	//
	while(ExitNow == 0)
	{
		buffer = (ADDRVALUE*)mem;
		expected_value = i - 1;

		for (word_ind = 0; word_ind < num_words; word_ind++)
		{
			actual_value = buffer;
			if (*actual_value != expected_value)
			{
				log_error(actual_value, *actual_value, expected_value);
			}
			*actual_value = i;

			buffer++;
		}

		i++;

		local_mem_errors = readLocalMemErrorsCounters();
		log_local_mem_errors(local_mem_errors);

		if (SleepTime > 0 && ExitNow == 0)
		{
			sleep(SleepTime);
		}
	}

	free(mem);

	local_mem_errors = stopLocalMemErrorsCounters();
	log_local_mem_errors(local_mem_errors);
}

void zero_one_test(void* mem, unsigned int num_bytes)
{
    ADDRVALUE expected_value = 0x0;
    ADDRVALUE* word_ind = NULL;
    unsigned char* startMem = (unsigned char*)mem;
    unsigned char* endMem = startMem + num_bytes;

    //
    // Start the HW counters for the LocalMemErrors
    //
    int local_mem_errors = startLocalMemErrorsCounters();
    if (!local_mem_errors)
    {
        char msg[] = "Cannot start LocalMemError counters.";
        log_message(msg);
        //ExitNow = 1;
        if (daemon_pid_file_exists())
        {
            daemon_pid_delete_file();
        }
    }

    //
    // The main loop
    //
    while(ExitNow == 0)
    {
    
    // 
    // Simulate error at random position and random iteration.
    //
#ifdef INJECT_ERR
        int riter = rand()%10 + 1;
        if (riter == 5) {
            unsigned int ruibytes;
            int ribytes = rand();
            memcpy(&ruibytes, &ribytes, sizeof(ribytes));
            ruibytes = (ruibytes%num_bytes);
            unsigned char *ptr_data = startMem + ruibytes;
            memset(ptr_data, 0x1, 1);
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask); 
            char msg[1000];
            snprintf(msg,1000,"Inject error at %p", ptr_data);
            log_message(msg);
        }
#endif
        
        for (word_ind = (ADDRVALUE*)startMem; word_ind < (ADDRVALUE*)endMem; word_ind++)
        {
           if (*word_ind != expected_value)
           {
               log_error(word_ind, *word_ind, expected_value);
           }
           *word_ind = ~expected_value;
        }
        expected_value = ~expected_value;

        local_mem_errors = readLocalMemErrorsCounters();
        log_local_mem_errors(local_mem_errors);

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }
    }

    free(mem);

    local_mem_errors = stopLocalMemErrorsCounters();
    log_local_mem_errors(local_mem_errors);
}

void random_pattern_test(void* mem, unsigned int num_bytes)
{
    ADDRVALUE expected_value;
    ADDRVALUE* word_ind = NULL;
    unsigned char* startMem = (unsigned char*)mem;
    unsigned char* endMem = startMem + num_bytes;

    //
    // Start the HW counters for the LocalMemErrors
    //
    int local_mem_errors = startLocalMemErrorsCounters();
    if (!local_mem_errors)
    {
        char msg[] = "Cannot start LocalMemError counters.";
        log_message(msg);
        //ExitNow = 1;
        //if (daemon_pid_file_exists())
        //{
        //    daemon_pid_delete_file();
        //}
    }

    memset(&expected_value, mem_pattern, sizeof(uint64_t));
    memset(mem, mem_pattern, num_bytes);
    
    //
    // The main loop
    //
    while(ExitNow == 0)
    {
        // 
        // Simulate error at random position and random iteration.
        //
#ifdef INJECT_ERR
        int riter = rand()%10 + 1;
        if (riter == 5) {
            unsigned int ruibytes;
            int ribytes = rand();
            memcpy(&ruibytes, &ribytes, sizeof(ribytes));
            ruibytes = (ruibytes%num_bytes);
            unsigned char *ptr_data = startMem + ruibytes;
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask); 
            *ptr_data = ~expected_value;
            char msg[1000];
            snprintf(msg,1000,"Inject error at %p", ptr_data);
            log_message(msg);
        }
#endif
        
        for (word_ind = (ADDRVALUE*)startMem; word_ind < (ADDRVALUE*)endMem; word_ind++)
        {
           if (*word_ind != expected_value)
           {
               log_error(word_ind, *word_ind, expected_value);
           }
           *word_ind = ~expected_value;
        }
        expected_value = ~expected_value;

        local_mem_errors = readLocalMemErrorsCounters();
        log_local_mem_errors(local_mem_errors);

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }
    }

    free(mem);

    local_mem_errors = stopLocalMemErrorsCounters();
    log_local_mem_errors(local_mem_errors);
}
void check_no_daemon_running()
{
	pid_t pid = daemon_pid_read_from_file();
	if (pid != 0)
	{
		fprintf(stderr, "It appears that an instance of this daemon is running because the %s pid file exists. If you are sure that there is no daemon running delete the pid file: %s.\n", PidFileName, PidFileName);
		exit(EXIT_FAILURE);
	}
}
void initialize_memory()
{
	while(!Mem && NumBytes > MEGA)
	{
		Mem = malloc(NumBytes);
		if (!Mem)
		{
			NumBytes -= MEGA*10;
		}
	}

	if (!Mem)
	{
		char msg[255];
		sprintf(msg, "ERROR_INFO,Cannot allocate %llu number of bytes.", NumBytes);
		log_message(msg);
		sleep(2);
		if (daemon_pid_file_exists())
		{
			daemon_pid_delete_file();
		}
		exit(EXIT_FAILURE);
	}
    memset(Mem, 0x0, NumBytes);
}

void sigterm_signal_handler(int signum)
{
	ExitNow = 1;
	log_message("STOP,SIGTERM");
}
void sigint_signal_handler(int signum)
{
	ExitNow = 1;
	log_message("STOP,SIGINT");
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

void daemon_pid_write_to_file(pid_t pid)
{
	FILE *f = fopen(PidFileName, "w+");
	if (!f)
	{
		fprintf(stderr, "ERROR: Cannot create a daemon pid file.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(f, "%d", pid);
	fclose(f);
}

unsigned char daemon_pid_file_exists()
{
	unsigned char is_file_exists = 0;

	FILE *f = fopen(PidFileName, "w");
	if (f)
	{
		is_file_exists = 1;
		fclose(f);
	}

	return is_file_exists;
}

pid_t daemon_pid_read_from_file()
{
	pid_t pid = 0;
	int fscanStatus = 0;
	FILE *f = fopen(PidFileName, "r");
	if (f)
	{
		int pid_int = 0;
		fscanStatus = fscanf(f, "%d", &pid_int);
		if (fscanStatus <= 0)
		{
		    fprintf(stderr, "ERROR: Cannot fscanf file %s for pid.\n", PidFileName);
		}
		pid = (pid_t)pid_int;
		fclose(f);
	}
	return pid;

}
void daemon_pid_delete_file()
{
	int delete_status = remove(PidFileName);
	if (delete_status != 0)
	{
		fprintf(stderr, "ERROR: Cannot delete the file %s. Probably no such file exists.\n", PidFileName);
	}
}

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


