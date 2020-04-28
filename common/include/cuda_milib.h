/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * cuda_milib.h
 ******************************************************/
#include <iostream>
#include <cstring>

#include <sys/types.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Utilities and system includes
#include <milib.h>

#ifndef _CUDA_MILIB_H_
#define _CUDA_MILIB_H_

#define CUDA_FREE(x)		if(x){gpuErrChk(cudaFree((x))); (x)=NULL;}
#define CUDA_GLOBAL_CHECK {gpuErrChk (cudaPeekAtLastError ()); gpuErrChk (cudaDeviceSynchronize ());}

#if defined(__CUDACC__)		// NVCC
	#define MY_ALIGN(n)		__align__(n)
#elif defined(__GNUC__)		// GCC
	#define MY_ALIGN(n)		__attribute__((aligned(n)))
#elif defined(_MSC_VER)		// MSVC
	#define MY_ALIGN(n)		__declspec(align(n))
#else
	#error "Please provide a definition for MY_ALIGN for your host compiler"
#endif
/* =================== CUDA related structures ====================== */
typedef struct gpu_config {
	int	n_threads;	// maxThreads allowed per block
	int	dev_count;	// number of cuda devices

	size_t	shmem;		// sh_mem per block
	size_t	free_mem;	// free memory on the card
	size_t	tot_mem;	// total memory on the card
	struct	cudaDeviceProp	dev_prop;	// device properties
	CUcontext cuc;	// context
} gpu_config;

/* ====================== Cuda Debug macros ======================== */
#define gpuErrChk(ans)		{gpuAssert((ans), \
__FILE__, __func__, __LINE__);}
#ifndef NDEBUG
	#define CudaDbgPrn(M, ...) {printf \
	("DevDBG:%s:%s:%d: " M "\n", \
	__FILE__, __func__, (int) __LINE__, ##__VA_ARGS__);}
	#define PRINT_HISTO(M,N)	{Dbg("========= start print_token_tracking ========="); print_token_tracking ((M), (N)); printf("========= end print_token_tracking =========\n");}
	#define DBG_EXIT			{Dbg ("Exiting..."); exit (1);}
	#define KER_RETURN			{if (!gTh) CudaDbgPrn ("Returning..."); return;}
#else
	#define CudaDbgPrn(M, ...) {void;}
	#define PRINT_HISTO(M,N)	{void;}
	#define DBG_EXIT			{void;}
	#KER_RETURN					{void;]
#endif

inline void gpuAssert (cudaError_t code, const char *file, const char *func, int line)
{
	bool abort = true;
	bool sig_raise = true;

	if (code != cudaSuccess) {
		fprintf(stderr,"CUDA call from file:%s func:%s %d: %s:%s failed\n", file, func, line, cudaGetErrorName(code), cudaGetErrorString(code));

		if (sig_raise) raise (SIGSEGV);
		if (abort) exit (code);
	}
}

inline unsigned maxThreads(uint64_cu th, const unsigned unroll)
{
	return ((th % unroll) ? (th / unroll + 1): (th / unroll));
}

inline bool isOdd (unsigned int i)
{
	if (0 != (i % 2)) return true;
	else return false;
}

/* =================== histogram structures ====================== */
// assumption: CHUNK_SIZE will always be > 16 so that spillover can be handled.
// #define CHUNK_SIZE				4112
// #define CHUNK_SIZE				8192
// #define CHUNK_SIZE				32  // For testing
#define STAT_MAX_WORD_SIZE			32
// statistically 99.999% of words are within 20 chars + 4 chars to ensure that next word doesn't impact the hashing function.  Extra 10 char (size = 30) is buffer - 2 chars are to ensure hashing boundary is not crossed
// RS_DEBUG - I don't handle unicode

// the next two defines are for cuda_MemCpy2D
#define NUM_ROWS					1

// base histogram info
typedef struct stream_info {
	unsigned char	*d_curr_buf;
	unsigned char	*h_read_buf;
	unsigned int	*d_loc;		// location of word
	size_t			pitch;		// pitch of d_loc
	size_t			*d_loc_maxsz;	// max sz of d_loc
	size_t			bufsiz;		// actual buf size read
} stream_info;

// mhash - modify this if the hashing needs to change
typedef struct MY_ALIGN (16) mhash_vals {
	unsigned char	str [2];
	uint32_t		mhash;
} mhash_vals;

typedef struct MY_ALIGN (16) words {
	unsigned char	str [STAT_MAX_WORD_SIZE + 2];
	unsigned int	len;
	uint32_t		num;
} words;

typedef struct MY_ALIGN (16) token {	// info on each word
	mhash_vals		mhv;			// hash for word
	words			wd;				// specific word
} token;

typedef struct token_tracking {		// token data
	bool			stopWord_applied; // stop word
	unsigned int	h_num_free;		// tokens free on host
	unsigned int	chunk_size;		// CHUNK_SIZE of this tt
	uint32_t		first_hash;		// first hash in chunk

	unsigned int	*d_num_free;	// tokens free on dev
	token			**d_data;		// CHUNK_SIZE on dev
	token			*h_data;		// cp of dev on host
	token			*d_tok_start;	// token in d_data

	token_tracking	*next;
	token_tracking	*prev;
} token_tracking;

typedef struct MY_ALIGN (16) tok_mhash {
	token			*tok;
	mhash_vals		mhv;
} tok_mhash;		// sort in shared mem

typedef struct math_results {
	unsigned int	list_len;
	uint32_t		h_tot_unique_wds;	// sum (words)
	unsigned int	h_tot_wds_pre_sw;	// sum (all num)
	uint32_t		h_tot_wds_pst_sw;	// sum (all num)
	double			std_dev;

	words		*d_num_list;	// sort here &
	words		*h_num_list;	// txfr to here

	words		avg;
	words		median;
	words		min;
	words		max;
} math_results;

/**
 * block_size: is a unit of tt.  = shmem / token
 * warp_size: is a single warp
 * if (data > block_size) then multiple tt
 **/
typedef struct MY_ALIGN (16) gpu_calc {		// for diff env
	dim3			block;
	unsigned int	block_size;	// shared memory / token sz
	unsigned int	warp_size;	// warp_size / token sz
} gpu_calc;

typedef struct all_bufs {		// all global variables
	gpu_config		gc;
	gpu_calc		def_gpu_calc;

	bool			IsS_wd;
	stream_info		st_info;
	mhash_vals		*d_sw_list;
	unsigned int	*d_sw_nos;

	token_tracking	*h_tcurr;	// current token CPU/tok GPU
	token_tracking	*h_ttrack;	// token CPU/tok GPU
	math_results	h_math;

	unsigned int	*d_wds;		// words read
	unsigned int	h_Swds;		// stop words read
} all_bufs;

/* ====================== Exported functions ======================== */
#undef __BEGIN_DECLS
#undef __END_DECLS

#ifdef __cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }

#else		// __cplusplus
#define __BEGIN_DECLS			/* empty */
#define __END_DECLS				/* empty */

#endif		// __cplusplus

__BEGIN_DECLS
unsigned int process_next_set_of_tokens (all_bufs *ab, ssize_t data_read);
void math_histogram (all_bufs *ab);
void cuda_free_everything (all_bufs *ab);

__device__ void cuda_strn_cpy (unsigned char *tgt, const unsigned char *src, const unsigned len);
__device__ unsigned cuda_string_len (unsigned char *str);
uint64_cu asc_to_uint64_cu (const char *txt);
__device__ __host__ uint32_cu cuda_int_div_rounded (uint64_cu num, const unsigned div);
__device__ __host__ bool cuda_AsciiIsAlpha (unsigned char c);
__device__ __host__ bool cuda_AsciiIsAlnum (unsigned char c);
__device__ __host__ bool cuda_AsciiIsAlnumApostrophe (unsigned char c);
__device__ __host__ void *cuda_MemCpy (void *pDst, void *pSrc, const uint32_t size);
__device__ __host__ void *cuda_MemSet (void *pDst, const uint8_t val, const uint32_t size);

int timeval_subtract (struct timeval *result, struct timeval *t2, struct timeval *t1);

__END_DECLS

#endif		// _CUDA_MILIB_H_
