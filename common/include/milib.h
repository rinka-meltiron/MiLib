/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * milib.h
 ******************************************************/

#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>


#ifndef _MILIB_H_
#define _MILIB_H_

// ERROR CODES
#define ERROR					-1
#define FAILED					0
#define	WRONG_ARGS				2
#define WRONG_PARAMS			3
#define FOUND_PATTERN			4
#define PATTERN_NOT_FOUND		5
#define SUCCESS					6
#define OUT_OF_MEMORY			7
#define MID_FILE				8
#define END_OF_FILE				9

#define AVERAGE_WORD			5

#define ONE_MB				(1*1024*1024*1024) // 1 MB space
// #define ONE_MB				(100*1024*1024) // 100 KB space
// #define ONE_MB				(100*1024) // 100 KB space
// #define ONE_MB					(100) // 100 bytes space
#define EIGHTY_PERCENT			(0.8)
#define SEVENTY_PERCENT			(0.7)
#define SIXTY_PERCENT			(0.6)
#define FIFTY_PERCENT			(0.5)
#define FORTY_PERCENT			(0.4)
#define THIRTY_PERCENT			(0.3)
#define TWENTY_PERCENT			(0.2)
#define TWO_PERCENT				(0.02)

#define FREE(x)				if(x){free((x)); (x)=NULL;}

typedef long int				int32_cu;
typedef long long int			int64_cu;
typedef unsigned long int		uint32_cu;
typedef unsigned long long int	uint64_cu;

typedef enum {FIRST_MATCH, ALL_MATCHES}			pattern_type;
typedef enum {ONE_CPU, ALL_CPU, GPU, BOTH}		cpu_type;
typedef enum {stream_pmatch, record_pmatch, \
				histogram, err}					functionality;

/* =================== record search STRUCTURES ===================== */
#define MAX_RECORDS		829137		// maximum records
#define MAX_PATTERN_SIZE	20		// maximum pattern size
#define ONE_LINE			80		// sizeof one line

// external
typedef struct input_to_gpu_interface {
	functionality		func;		// func. to implement
	cpu_type			cpu;
	struct pattern_st	*pat;		// for pattern matches
	struct pattern_st	*gpu_pat;	// pattern on GPU
	struct record_node	*all_nodes;	// all the records
	struct gpu_config	*gpu_cfg;	// configuration info
	struct record_node	*gpu_nodes;	// records in GPU
	bool				*result;	// results on host
	bool			gpu_result [];	// results on GPU
} input_to_gpu_interface;


// this is the data inputed
typedef struct record_node {
	uint64_cu	rec_id;		// 0 to whatever
	int		len;			// length of data
	char	data [ONE_LINE];	// match the data
} record_node;

typedef struct pattern_st {
	char	*pattern;
	int		pat_sz;
} pattern_st;

// this is the match result
typedef struct result_node {
	int		rec_id;			// the match found
	struct result_node	*last;	// the last node in the list
	struct result_node	*next;	// the next node in the list
} result_node;

/* ================== Pattern search STRUCTURES ==================== */
typedef struct cpu_info {
	cpu_type		type;		// default type CPU
	int				pid;		// process id
	int				tot_thread;	// total threads
	struct thpool_	*thpool;	// the thread pool
} cpu_info;

// these are the chunks of string based on the number of threads possible
// this will be an array of the tid - thread id
typedef struct str_chunk {
	char			*str_start;	// the start of each chunk
	uint64_cu		id_start;	// loc. of start of chunk in full string
	uint64_cu		chunk_len;	// the length of each chunk of data
	int				thread_id;	// id of the thread (0 to n-1) NOT 1-n
	struct pat_info	*pt;		// pattern data
} str_chunk;

// the locations where the data was found
typedef struct found_loc {
	uint64_cu	loc;			// location
	struct found_loc	*next;	// next location
} found_loc;

// main pattern information
typedef struct pat_info {	// storage for pattern
	char		*str;		// the full buffer - rmd as ONLY file
	uint64_cu	prev_read;	// pointer at the end of the last read
	uint64_cu	curr_read;	// pointer at the end of the current read
	uint64_cu	str_sz;		// size read on the current read
	cpu_info	*cpu;		// the type of CPU & threads supported
	char		*pat_str;	// string to be matched
	int			pat_len;	// length of string to be matched
	char		*fname;		// file name
	FILE		*input;		// file to be read for data
	uint64_cu	fsize;		// size of the file

	pattern_type	tp;		// FIRST_MATCH/ALL_MATCHES
	found_loc	*fnd;		// all
	found_loc	*end;		// the is the end of the list fnd;
	str_chunk	*str_c;		// the chunks of the full buffer
} pat_info;

// Alignment structures
// Struct alignment is handled differently between the CUDA compiler and other

/*** usage
 * typedef struct ALIGN(16) _SomeTypeStruct
 * {int i;} SomeType;
 ***/
#if defined(__CUDACC__) // NVCC
	#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
	#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
	#define ALIGN(n) __declspec(align(n))
#else
	#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

/*******
 * threads
 ******/
#define MAX_NANOSEC 999999999
#define CEIL(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))

/* ====================== Threadpool STRUCTURES ======================== */

/* Binary semaphore */
typedef struct bsem {
	pthread_mutex_t mutex;
	pthread_cond_t   cond;
	int v;
} bsem;

/* Job */
typedef struct job{
	struct job *prev;					/* pointer to previous job   */
	void *(*function)(void* arg);		/* function pointer          */
	void *arg;							/* function's argument       */
} job;

/* Job queue */
typedef struct jobqueue{
	pthread_mutex_t rwmutex;             /* used for queue r/w access */
	job  *front;                         /* pointer to front of queue */
	job  *rear;                          /* pointer to rear  of queue */
	bsem *has_jobs;                      /* flag as binary semaphore  */
	int   len;                           /* number of jobs in queue   */
} jobqueue;

/* Thread */
typedef struct thread{
	int       id;                        /* friendly id               */
	pthread_t pthread;                   /* pointer to actual thread  */
	struct thpool_* thpool_p;            /* access to thpool          */
} thread;

/* Threadpool */
typedef struct thpool_{
	thread**   threads;					// pointer to threads
	volatile int num_threads_alive;		// threads currently alive
	volatile int num_threads_working;	// threads currently working
	pthread_mutex_t	thcount_lock;		// used for thread count etc
	pthread_mutex_t	thpool_wait_lock;	// use in thpool_wait

	jobqueue*  jobqueue_p;				// pointer to the job queue
} thpool_;


/* ====================== Debug macros ======================== */
#ifdef NDEBUG
	#define Dbg(M, ...)
	#define DEBUG_CODE(M)
#else
	#define Dbg(M, ...) fprintf(stderr, "DEBUG %s:%s:%d: " M "\n", __FILE__, \
	__func__, __LINE__, ##__VA_ARGS__)
	#define DEBUG_CODE(M)	(M)
#endif	// NDEBUG

/*******************************
#define log_warn(M, ...)		fprintf(stderr, "[WARN] (%s:%s:%d: errno: %s) " M "\n", __FILE__, __func__, __LINE__, clean_errno(), ##__VA_ARGS__)
#define log_info(M, ...)		fprintf(stderr, "[INFO] (%s:%s:%d) " M "\n", __FILE__, __func__, __LINE__, ##__VA_ARGS__)
#define sentinel(M, ...)		{ log_err(M, ##__VA_ARGS__); errno=0; goto error; }
#define check_debug(A, M, ...)  if(!(A)) { debug(M, ##__VA_ARGS__); errno=0; goto error; }
********************************/
#define clean_errno()		   (errno == 0 ? "None" : strerror(errno))

#define log_err(M, ...)		 fprintf(stderr, "[ERROR] (%s:%s:%d: errno: %s) " M \
					"\n", __FILE__, __func__, __LINE__, clean_errno(), ##__VA_ARGS__)

#define check(A, M, ...)		if((A)) {log_err(M, ##__VA_ARGS__); errno=0; goto error; }

#define check_mem(A)			check(!(A), "Out of memory.")

#endif	// _MILIB_H_

