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
 Name        : MemoryReliability.c
 Authors     : Ferad Zyulkyarov, Kai Keller, Leonardo Bautista-Gomez
 Version     :
 Copyright   :
 Description : A daemon which tests the memory for errors.

 This file implements a simple memory test tool which can run as foreground
 process or as a daemon.

 The executable can be used to:
 1) Start daemon
 2) Stop daemon
 3) Start foreground process

 Before starting a process as a daemon we first check if the daemon pid file
 exists and contains a valid value. The pid file name is stored in
 PidFileName. If such file exists we do not create a new daemon and print
 an error message.

 If the pid file does not exist, we create one and store in it the pid of the
 daemon process. Then we initialize the memory and run the daemon.

 The daemon scans the memory for errors and logs the error if it finds one.

 The daemon stops when it receives a SIGTERM signal that can be send either
 by calling "kill daemon_pid" or calling this executable with the "-c"
 argument. In the latter case, we read the pid of the daemon from the
 pid file, send the SIGTERM signal and delete the pid file.

 The client also intercepts and handles the SIGTERM signal and exists gently
 when such signal is received.

 ============================================================================
 Adaptation for Mare Nostrum IV, Oct 2017.
 Author      : Kai Keller
 ============================================================================
 */

#include "MemoryReliability_decl.h"

//
// The main daemon task
//

int main(int argc, char* argv[])
{
	bool is_init = parse_arguments(argc, argv);
	if (!is_init)
	{
		print_usage(argv[0]);
		return EXIT_FAILURE;
	}

	if (IsDaemonStop)
	{
		stop_daemon();
	}
	else if (IsDaemonStart)
	{
		check_no_daemon_running();
		start_daemon();
	}
	else
	{
		start_client();
	}

	return EXIT_SUCCESS;
}

bool parse_arguments(int argc, char* argv[])
{
	int i = 1;
	unsigned int mega = 0;

    srand(time(NULL));
    unsigned char BinExpo = rand()%4;
    SleepTime = 0x0001<<BinExpo;
    mem_pattern = (unsigned char)rand()%256; // generate byte pattern

	memset(HostName, 0, 255);
	gethostname(HostName, 254);

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-c") == 0)
		{
			IsDaemonStop = 1;
		}
		if (strcmp(argv[i], "-d") == 0)
		{
			IsDaemonStart = 1;
		}
        if (strcmp(argv[i], "--cpu") == 0)
        {
            i++;
            CheckCPU = 1;

            if (strcmp(argv[i], "-m") == 0)
            {
                i++;
                unsigned int mega = (unsigned int)atoi(argv[i]);
                NumBytesCPU = (unsigned long long)mega * (unsigned long long)MEGA;
                if (NumBytesCPU%8 != 0) {
                    printf("The number of Bytes has to be a multiple of 8");
                }
            }
        }
        if (strcmp(argv[i], "--gpu") == 0)
        {
            i++;
            CheckGPU = 1;

            if (strcmp(argv[i], "-m") == 0)
            {
                i++;
                unsigned int mega = (unsigned int)atoi(argv[i]);
                NumBytesGPU = (unsigned long long)mega * (unsigned long long)MEGA;
                if (NumBytesGPU%8 != 0) {
                    printf("The number of Bytes has to be a multiple of 8");
                }
            }
        }
		if (strcmp(argv[i], "-o") == 0)
		{
			i++;
			strcpy(OutFile, argv[i]);
		}
		if (strcmp(argv[i], "-e") == 0)
		{
			i++;
			strcpy(ErrFile, argv[i]);
		}
		if (strcmp(argv[i], "-s") == 0)
		{
			i++;
			SleepTime = (unsigned int)atoi(argv[i]);
		}
		if (strcmp(argv[i], "-wn") == 0)
		{
			i++;
			WarningRate = (unsigned int)atoi(argv[i]);
		}
		if (strcmp(argv[i], "-wf") == 0)
		{
			i++;
			strcpy(WarningFile, argv[i]);
		}
	}

	if (!IsDaemonStop)
	{
		printf("Host name: %s\n", HostName);
        if (CheckCPU)
        {
            printf("CPU Memory size: %llu bytes = %llu MB\n", NumBytesCPU, NumBytesCPU/MEGA);
            printf("CPU Memory pattern (hex): 0x%" PRIxPTR "\n", mem_pattern);
        }
        if (CheckGPU)
        {
            printf("GPU Memory size: %llu bytes = %llu MB\n", NumBytesGPU, NumBytesGPU/MEGA);
            printf("GPU Memory pattern (hex): 0x%" PRIxPTR "\n", mem_pattern);
        }
		printf("Log file: %s\n", OutFile);
		printf("Error log file: %s\n", ErrFile);
		printf("Sleep time: %u\n", SleepTime);
		printf("Warning rate: %u\n", WarningRate);
		printf("Warning file: %s\n", WarningFile);
	}

	bool is_success = (CheckCPU && NumBytesCPU != 0) || (CheckGPU && NumBytesGPU != 0) || (IsDaemonStop);

	if ((WarningRate > 0 && strlen(WarningFile) == 0) || (WarningRate == 0 && strlen(WarningFile) > 0))
	{
		is_success = 0;
		printf("Error with the provided command line arguments. Make sure that both -wn and -wf arguments are provided and correct.\n");
	}
	return is_success;
}

void print_usage(char* program_name)
{
	printf("Memory scanner by Ferad Zyulkyarov\n");
	printf("This program starts a daemon for memory test or stops an already running daemon.\n");
	printf("Options: [-d] [-c] [--cpu [-m]] [--gpu [-m]] [-o] [-s] [-wn] [-wf]\n");
	printf("    -d: start as daemon. If not passed will start as a foreground process.\n");
	printf("    -c: stop the daemon.\n");
    printf("    --cpu: check CPU memory.\n");
    printf("    -m: the size of the CPU memory to test in MB.\n");
    printf("    --gpu: check GPU memory.\n");
    printf("    -m: the size of the GPU memory to test in MB.\n");
	printf("    -o: log file name [default=%s].\n", OutFile);
	printf("    -e: error file name [default=%s].\n", ErrFile);
	printf("    -s: sleep time in seconds [default=%u].\n", SleepTime);
	printf("    -wn: warning rate [default=%u].\n", WarningRate);
	printf("    -wf: warning file [default=%s].\n", WarningFile);
	printf("Example to start a daemon:\n");
	printf("    %s -d --cpu -m 1024 -s 10 -o full_path_to_logfile.log -- run as a daemon, test 1024MB of CPU memory, and sleep for 10sec, write logs to logfile.log.\n", program_name);
	printf("    %s -d --cpu -m 1024 -s 10 -o full_path_to_logfile.log -wn 15 -wf full_path_to_warningfile.txt -- run as a daemon, test 1024MB of CPU memory, and sleep for 10sec, write logs to logfile.log, if there are more than (-wn) 10 errors detected write the name of the host to file warning.txt\n", program_name);
    printf("    %s -d --cpu -m 1024 --gpu -m 2048 -s 5 -o full_path_to_logfile.log -wn 15 -wf full_path_to_warningfile.txt -- run as a daemon, test 1024MB of CPU memory, test 2048MB of GPU memory, and sleep for 5sec, write logs to logfile.log, if there are more than (-wn) 10 errors detected write the name of the host to file warning.txt\n", program_name);
    printf("Example to stop a daemon: \n");
	printf("    %s -c\n", program_name);
	printf("Example to run as a foreground process:\n");
	printf("    %s --cpu -m 1024 -s 10 -- run as a daemon, test 1024MB of memory, and sleep for 10sec.\n", program_name);
}

void start_daemon()
{
	char start_msg[255];
	//
	// Fork child process
	//
	pid_t pid = fork();

	if (pid < 0)
	{
		fprintf(stderr, "ERROR: Cannot create child process.\n");
		exit(EXIT_FAILURE);
	}

	if (pid > 0)
	{
		//
		// Write the pid of the daemon to the pid file
		//
		daemon_pid_write_to_file(pid);
        printf("forked PID: %i. Parent process exits now.\n", (int) pid); /*KAI*/

		//
		// Child process created exit parent process.
		//
		exit(EXIT_SUCCESS);
	}

	//
	// Set file permissions for files created by our child process
	// This requires sudo.
	//
	// umask(S_IRWXO|S_IRWXO|S_IRWXU);

	//
	// Create session for our new process.
	//
	pid_t sid = setsid();
	if (sid < 0)
	{
		fprintf(stderr, "ERROR_INFO,Cannot create session for the child process.\n");
		log_message("ERROR_INFO,Cannot create session for the child process.");
		exit(EXIT_FAILURE);
	}

	//
	// Change the current working directory
	//
	//if ((chdir("/")) < 0)
	//{
	//	fprintf(stderr, "ERROR_INFO,Cannot change the working directory.\n");
	//	log_message("ERROR_INFO,Cannot change the working directory.");
	//	exit(EXIT_FAILURE);
	//}
    sprintf(start_msg, "START, SLEEP TIME (SECONDS): %u", SleepTime);
    log_message(start_msg);
    if (CheckCPU)
    {
        sprintf(start_msg, "INFO, ALLOCATED CPU MEMORY (BYTES): %llu, PATTERN (HEX): 0x%" PRIxPTR, NumBytesCPU, mem_pattern);
        log_message(start_msg);
    }
    if (CheckGPU)
    {
        sprintf(start_msg, "INFO, ALLOCATED GPU MEMORY (BYTES): %llu, PATTERN (HEX): 0x%" PRIxPTR , NumBytesGPU, mem_pattern);
        log_message(start_msg);
    }
	printf("Daemon started\n");

	//
	// Close all standard file descriptors
	//
	fclose(stdin);
	fclose(stdout);
	fclose(stderr);

	//
	// Install signal handler that responds to kill [pid] from the command line.
	//
	signal(SIGTERM, sigterm_signal_handler);

	//
	// Ignore signal when terminal session is closed. This keeps the
	// daemon alive when the user closes the terminal session.
	//
	signal(SIGHUP, SIG_IGN);

    // Initialize the memory before entering the daemon loop
    if (CheckCPU)
        initialize_cpu_memory();
    if (CheckGPU)
        initialize_gpu_memory();

	//
	// The daemon loop.
	//
    memory_test_loop(RANDOM);

    if (CheckCPU)
        free_cpu_memory();
    if (CheckGPU)
        free_gpu_memory();

	exit(EXIT_SUCCESS);
}

void stop_daemon()
{
	pid_t pid = daemon_pid_read_from_file();
	int kill_status = 0;
	if (pid == 0)
	{
		fprintf(stderr, "ERROR: It seems that there is no daemon running. Please, check that %s file exists and contains a valid pid value. If you are sure that the daemon is running, please, kill it manually.\n", PidFileName);
	}
	else
	{
		kill_status = kill(pid, SIGTERM);
		if (kill_status == 0)
		{
			printf("The daemon stopped.\n");

			//
			// Delete the daemon pid file.
			//
			daemon_pid_delete_file();
		}
		else
		{
			fprintf(stderr, "ERROR: Failed to stop the daemon with pid %d.\n", pid);
		}
	}
}

void start_client()
{
	char start_msg[255];

	//
	// Install signal handler that responds to kill [pid] from the command line.
	//
	signal(SIGTERM, sigterm_signal_handler);

	//
	// Install signal handler that responds to Ctrl+C interrupt
	//
	signal(SIGINT, sigint_signal_handler);

	printf("Client started...\n");

    sprintf(start_msg, "START, SLEEP TIME (SECONDS): %u", SleepTime);
    log_message(start_msg);
    if (CheckCPU)
    {
        sprintf(start_msg, "INFO, ALLOCATED CPU MEMORY (BYTES): %llu, PATTERN (HEX): 0x%" PRIxPTR, NumBytesCPU, mem_pattern);
        log_message(start_msg);
    }
    if (CheckGPU)
    {
        sprintf(start_msg, "INFO, ALLOCATED GPU MEMORY (BYTES): %llu, PATTERN (HEX): 0x%" PRIxPTR , NumBytesGPU, mem_pattern);
        log_message(start_msg);
    }

    if (CheckCPU)
        initialize_cpu_memory();
    if (CheckGPU)
        initialize_gpu_memory();

    //
    // The daemon loop.
    //
    memory_test_loop(RANDOM);

    if (CheckCPU)
        free_cpu_memory();
    if (CheckGPU)
        free_gpu_memory();
}

