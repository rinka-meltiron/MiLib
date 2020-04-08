//-----------------------------------------------------------------------------
// The MurmurHash3 function was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Origin:
// https://en.wikipedia.org/wiki/MurmurHash
// https://github.com/aappleby/smhasher
// https://github.com/PeterScott/murmur3

#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <execinfo.h>

#include <cuda_milib.h>

/*********************  Extern Library Functions  ************************/
extern unsigned int get_d_val (unsigned int *d_val);

/*********************  Internal Funcs  *********************/
static __global__ void K_dummy (void);
static void gpu_configuration (gpu_config *gc);

/****************  Exported Functions  ********************/
uint64_cu asc_to_uint64_cu (const char *txt);
int timeval_subtract (struct timeval *result, struct timeval *t2, struct timeval *t1);
void histo_time_taken (struct timeval *tvDiff, struct timeval *t2, struct timeval *t1, const char *messg);
__device__ __host__ uint32_cu cuda_int_div_rounded (uint64_cu num, const unsigned div);
void set_ab_values (all_bufs *ab);
void set_to_zero (unsigned int *d_val);
unsigned int get_d_val (unsigned int *d_val);

__device__ void cuda_strn_cpy (unsigned char *tgt, const unsigned char *src, const unsigned len);
__device__ void cuda_swap_mem (tok_mhash *d_mem1, tok_mhash *d_mem2);
__device__ unsigned cuda_cuda_string_len (unsigned char *str);
__device__ __host__ bool cuda_AsciiIsAlpha (unsigned char c);
__device__ __host__ bool cuda_AsciiIsAlnum (unsigned char c);
__device__ __host__ bool cuda_AsciiIsAlnumApostrophe (unsigned char c);

bool __device__ __host__ isGreater (const mhash_vals *curr, const mhash_vals *next);
bool __device__ __host__ isEqual (const mhash_vals *curr, const mhash_vals *next);

__device__ void cuda_acquire_semaphore (volatile int *lock);
__device__ void cuda_release_semaphore (volatile int *lock);

// get the configuration information from the CUDA card
// http://cuda-programming.blogspot.in/2013/01/how-to-query-to-devices-in-cuda-cc.html
static void gpu_configuration (gpu_config *gc)
{
	int i = 0;
	gpuErrChk (cudaDeviceReset ());		// reset device

	gpuErrChk (cudaGetDeviceCount (&gc -> dev_count));
	Dbg("Device count %d", gc -> dev_count);

	// gc -> dev_prqop = malloc (sizeof (cudaDeviceProp) * gc -> dev_count);
	// tbd for multiple devices only.

	gpuErrChk (cudaGetDeviceProperties (&(gc -> dev_prop), i));
	gc -> n_threads = gc -> dev_prop.maxThreadsPerBlock;

	gc -> shmem = gc -> dev_prop.sharedMemPerBlock;
	gpuErrChk (cudaMemGetInfo (&(gc -> free_mem), &(gc -> tot_mem)));

	Dbg ("Dev prop name: %s, tot_mem: %u sharedMemPerBlock %u\nwarpSize %d maxThreadsPerBlock %d\nmaxthreads per mprocessor %d",
	gc -> dev_prop.name, (unsigned) gc -> dev_prop.totalGlobalMem,
	(unsigned) gc -> dev_prop.sharedMemPerBlock,
	gc -> dev_prop.warpSize, gc -> dev_prop.maxThreadsPerBlock,
	gc -> dev_prop.maxThreadsPerMultiProcessor);

	K_dummy <<<1,1>>> ();
	CUDA_GLOBAL_CHECK;
}

static void create_buffer (all_bufs *ab);
static void create_buffer (all_bufs *ab)
{
	ab -> st_info.bufsiz = (size_t) ab -> gc.tot_mem / 8;
	/* 1/8th GPU mem because:
	 * 1 buffer for stream,		3 buffers for token
	 * 1 scratchpad,			2 operating area
	 * 1 for spare
	 */
	size_t	h_loc_maxsz = ab -> st_info.bufsiz / 3;	// 3 chars per word

	ab -> st_info.h_read_buf = (unsigned char *) malloc (ab -> st_info.bufsiz * sizeof (unsigned char));
	check_mem (ab -> st_info.h_read_buf);
	memset (ab -> st_info.h_read_buf, '\0', ab -> st_info.bufsiz * sizeof (unsigned char));

	ab -> h_tcurr = NULL;
	ab -> h_ttrack = NULL;
	ab -> IsS_wd = false;

	gpuErrChk (cudaMalloc (&(ab -> st_info.d_curr_buf), ab -> st_info.bufsiz * sizeof (unsigned char)));

	gpuErrChk (cudaMalloc (&(ab -> st_info.d_loc_maxsz), sizeof (size_t)));
	gpuErrChk (cudaMemcpy (ab -> st_info.d_loc_maxsz, &h_loc_maxsz, sizeof (size_t), cudaMemcpyHostToDevice));

	gpuErrChk (cudaMalloc (&(ab -> st_info.d_loc), sizeof (size_t) * h_loc_maxsz));
	gpuErrChk (cudaMemset (ab -> st_info.d_loc, '\0', sizeof (size_t) * h_loc_maxsz));

	gpuErrChk (cudaMalloc (&(ab -> d_wds), sizeof (unsigned int)));
	gpuErrChk (cudaMemset (ab -> d_wds, '\0', sizeof (unsigned int)));
	return;

error:
	exit (1);
}

/***
 * in general:
 * block_size is combo of gc.shmem / sizeof (token), maxThreadsPerBlock, warp
 * fyi: sizeof (token) == 64, sizeof (tok_mhash) == 32, sizeof (token_tracking) 64
 * warp_size == 32 based on card.  TBD: Check for future warp sizes RS_DEBUG
 *
 * block based on warp size & grid based on total size
 ***/
void set_ab_values (all_bufs *ab)
{
	memset (ab, '\0', sizeof (all_bufs));
	gpu_configuration (&(ab -> gc));	// configure the GPU
	create_buffer (ab);

	ab -> def_gpu_calc.warp_size = ((2 > ab -> gc.dev_prop.major) ? 16 : 32);	// > 2 for Fermi and above cards

	int block_size = ab -> gc.shmem / sizeof (tok_mhash);	// shmem is max
	if (ab -> gc.dev_prop.maxThreadsPerBlock < block_size) block_size = ab -> gc.dev_prop.maxThreadsPerBlock;
	block_size = (block_size / ab -> def_gpu_calc.warp_size) * ab -> def_gpu_calc.warp_size;		// multiple of warp size
	block_size /= 8;	// 8 chunks in one block for sort_merge_histo_cross

	ab -> def_gpu_calc.block_size = block_size;

	dim3 block (ab -> def_gpu_calc.block_size);	// block based on block_size
	ab -> def_gpu_calc.block = block;

	setvbuf (stdin, NULL, _IOFBF, 0);	// buffered read
}

static __global__ void K_dummy (void)
{
	unsigned int i;

	i = WRONG_ARGS + WRONG_ARGS;	// just running any code to warm the card up.
	i++;
}

uint64_cu asc_to_uint64_cu (const char *txt)
{
	 uint64_cu num = 0;
	 for (; ((*txt) && (*txt != '.')); txt++) {
		 char digit = *txt - '0';
		 num = (num * 10) + digit;
	 }

	 return num;
}

#define ONE_MIL			1000000
int timeval_subtract (struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + ONE_MIL * t2->tv_sec) - (t1->tv_usec + ONE_MIL * t1->tv_sec);
	result->tv_sec = diff / ONE_MIL;
	result->tv_usec = diff % ONE_MIL;

	return (diff < 0);
}

void histo_time_taken (struct timeval *tvDiff, struct timeval *t2, struct timeval *t1, const char *messg)
{
	timeval_subtract (tvDiff, t2, t1);
	printf ("%s %ld.%06ld sec\n", messg, tvDiff -> tv_sec, tvDiff -> tv_usec);
}

void set_to_zero (unsigned int *d_val)
{
	unsigned int i = 0;
	gpuErrChk (cudaMemcpy (d_val, &i, sizeof (unsigned int), cudaMemcpyHostToDevice));
}

unsigned int get_d_val (unsigned int *d_val)
{
	unsigned int tmp_wds = 0;
	gpuErrChk (cudaMemcpy (&tmp_wds, d_val, sizeof (unsigned int), cudaMemcpyDeviceToHost));

	return tmp_wds;
}

/*********************  Internal Functions  ************************/
__device__ __host__ uint32_cu cuda_int_div_rounded (uint64_cu num, const unsigned div)
{
	if (0 == div) return FAILED;
	if ((num / div) * div == num)
		return num / div;
	else
		return (num / div + 1);
}

__device__ void cuda_strn_cpy (unsigned char *tgt, const unsigned char *src, const unsigned len)
{
	unsigned i;
	// CudaDbgPrn ("len %u", len);
	for (i = 0; i < len && i < STAT_MAX_WORD_SIZE; i++) tgt [i] = src [i];
	tgt [len] = (unsigned char)  '\0';
}

__device__ unsigned cuda_string_len (unsigned char *str)
{
	unsigned i = (unsigned) 0;
	unsigned char *s = str;
	for (; s && *s; ++s) ++i;

	if (STAT_MAX_WORD_SIZE <= i) i = STAT_MAX_WORD_SIZE;
	return (i);
}

__device__ __host__ bool cuda_AsciiIsAlpha (unsigned char c)
{
	// return ((unsigned int) (c | ('A' ^ 'a') - 'a' <= 'z' - 'a'));
	if (((unsigned int) (c | 32) - 97) < 26U) return true;
	return false;
}

__device__ __host__ bool cuda_AsciiIsAlnum (unsigned char c)
{
	register unsigned char d = c & ~0x20;

	if ((c >= '0' && c <= '9') || (d >= 'A' && d  <= 'Z')) return true;
	return false;
}

__device__ __host__ bool cuda_AsciiIsAlnumApostrophe (unsigned char c)
{
	register unsigned char d = c & ~0x20;

	if ((c >= '0' && c <= '9') || (d >= 'A' && d  <= 'Z') || c == '\'') return true;
	else return false;
}

// RS_DEBUG currently tok_mhash. should be made generic & put in cuda_milib.h
__device__ void cuda_swap_mem (tok_mhash *d_mem1, tok_mhash *d_mem2)
{
	register tok_mhash tmp_mem;

	cuda_MemSet (&tmp_mem, (uint8_t) 0x00, (uint32_t) sizeof (tok_mhash));
	// CudaDbgPrn ("Before: d_mem1:mhash %u str %.2s|d_mem2:mhash %u str %.2s", (unsigned) d_mem1 -> mhv.mhash, d_mem1 -> mhv.str, (unsigned) d_mem2 -> mhv.mhash, d_mem2 -> mhv.str);

	// d_mem1 -> tmp_mem
	cuda_MemCpy (&(tmp_mem.mhv), &(d_mem1 -> mhv), sizeof (mhash_vals));
	tmp_mem.tok = (token *) d_mem1 -> tok;
	// CudaDbgPrn ("(tmp_mem<-d_mem1) mhash %u str %.2s|mhash %u str %.2s", (unsigned) tmp_mem.mhv.mhash, tmp_mem.mhv.str, (unsigned) d_mem1 -> mhv.mhash, d_mem1 -> mhv.str);

	// d_mem2 -> d_mem1
	cuda_MemCpy (&(d_mem1 -> mhv), &(d_mem2 -> mhv), sizeof (mhash_vals));
	d_mem1 -> tok = (token *) d_mem2 -> tok;
	__threadfence ();
	// CudaDbgPrn ("(d_mem1<-d_mem2) d_mem1 mhash %u str %.2s|d_mem2 mhash %u str %.2s", (unsigned) d_mem1 -> mhv.mhash, d_mem1 -> mhv.str, (unsigned) d_mem2 -> mhv.mhash, d_mem2 -> mhv.str);

	// tmp_mem -> d_mem2
	cuda_MemCpy (&(d_mem2 -> mhv), &(tmp_mem.mhv), sizeof (mhash_vals));
	d_mem2 -> tok = (token *) tmp_mem.tok;
	// CudaDbgPrn ("(d_mem2<-tmp_mem):mhash %u str %.2s|d_mem2:mhash %u str %.2s", (unsigned) d_mem1 -> mhv.mhash, d_mem1 -> mhv.str, (unsigned) d_mem2 -> mhv.mhash, d_mem2 -> mhv.str);
	// CudaDbgPrn ("After: d_mem1:mhash %u str %.2s|d_mem2:mhash %u str %.2s", (unsigned) d_mem1 -> mhv.mhash, d_mem1 -> mhv.str, (unsigned) d_mem2 -> mhv.mhash, d_mem2 -> mhv.str);
}

__device__ __host__ void *cuda_MemCpy (void *pDst,void *pSrc, const uint32_t size)
{
	typedef struct MY_ALIGN (32) memCpy_struct {
		uint32_t	leftAlign, rightAlign;
		uint32_t	sz;
		uint64_t	lastWord, currWord;
	} memCpy_struct;

	assert (pSrc);
	assert (pDst);

	memCpy_struct	mcp;
	uint64_t	*pSrc64, *pDst64;
	uint8_t		*p_Src8, *p_Dst8;

	p_Src8	= (uint8_t*)(pSrc);
	// CudaDbgPrn ("pSrc: %p\t\tpDst: %p", pSrc, pDst);
	// CudaDbgPrn ("pSrc: %.20s\t\tpDst: %s\t\tsize %u", (char *) pSrc, (char *) pDst, (unsigned) size);

	p_Dst8	= (uint8_t*) (pDst);
	mcp.sz	= size;
	/* copy byte by byte till the source first alignment this is necessarily to
	 * ensure we do not try to access data which is before the source buffer,
	 *  hence not ours.
	 */
	while(((uintptr_t) p_Src8 & 7) && mcp.sz) { // (pSrc mod 8) > 0 and size > 0
		*p_Dst8++ = *p_Src8++;
		mcp.sz--;
	}

	/* align destination (possibly disaligning source)*/
	while(((uintptr_t) p_Dst8 & 7) && mcp.sz) { // (pDst mod 8) > 0 and size > 0
		*p_Dst8++ = *p_Src8++;
		mcp.sz--;
	}

	/* dest is aligned and source is not necessarily aligned */
	mcp.leftAlign = (uint32_t)(((uintptr_t) p_Src8 & 7) << 3); // leftAlign = (pSrc mod 8)*8
	mcp.rightAlign = 64 - mcp.leftAlign;


	if (mcp.leftAlign == 0) {
		/* source is also aligned */
		pSrc64 = (uint64_t*) (p_Src8);
		pDst64 = (uint64_t*) (p_Dst8);
		while (mcp.sz >> 3) { /* size >= 8 */
			*pDst64++ = *pSrc64++;
			mcp.sz -= 8;
		}
		p_Src8 = (uint8_t*) (pSrc64);
		p_Dst8 = (uint8_t*) (pDst64);
	}
	else {
		/* source is not aligned (destination is aligned)*/
		pSrc64 = (uint64_t*) (p_Src8 - (mcp.leftAlign >> 3));
		pDst64 = (uint64_t*) (p_Dst8);
		mcp.lastWord = *pSrc64++;
		while(mcp.sz >> 4) { /* size >= 16 */
			mcp.currWord = *pSrc64;
			*pDst64 = (mcp.lastWord << mcp.leftAlign) | (mcp.currWord >> mcp.rightAlign);
			mcp.lastWord = mcp.currWord;
			pSrc64++;
			pDst64++;
			mcp.sz -= 8;
		}
		p_Dst8 = (uint8_t*) (pDst64);
		p_Src8 = (uint8_t*) (pSrc64) - 8 + (mcp.leftAlign >> 3);
	}

	/* complete the left overs */
	while (mcp.sz--) *p_Dst8++ = *p_Src8++;

	return pDst;
}

__device__ __host__ void *cuda_MemSet (void* pDst, const uint8_t val, const uint32_t size)
{
	typedef struct memSet_struct {
		uint32_t	sz;
		uint64_t	val64;
	} memSet_struct;

	const uint8_t		vl = val;
	uint64_t	*pDst64;
	uint8_t		*p_Dst8;

	register memSet_struct	mst;
	p_Dst8 = (uint8_t*) (pDst);
	mst.sz = size;

	/* generate four 8-bit val's in 32-bit container */
	mst.val64 = (uint64_t) vl;
	mst.val64 |= (mst.val64 << 8);
	mst.val64 |= (mst.val64 << 16);
	mst.val64 |= (mst.val64 << 24);
	mst.val64 |= (mst.val64 << 32);

	/* align destination to 64 */
	while(((uintptr_t) p_Dst8 & 7) && mst.sz) {	// (pDst mod 8) > 0 and size > 0
		*p_Dst8++ = vl;
		mst.sz--;
	}

	pDst64 = (uint64_t*) (p_Dst8);				// 	64-bit chunks
	while (mst.sz >> 4) {						// size >= 8
		*pDst64++ = mst.val64;
		mst.sz -= 8;
	}

	p_Dst8 = (uint8_t*) (pDst64);				// 	complete the leftovers
	while (mst.sz--) *p_Dst8++ = vl;

	return pDst;
}

bool __device__ __host__ isGreater (const mhash_vals *curr, const mhash_vals *next)
{
	if (curr -> mhash) {
		// CudaDbgPrn ("mhash: %u/%u: curr: %.2s next %.2s", (unsigned) curr -> mhash, (unsigned) next -> mhash, curr -> str, next -> str);
		// CudaDbgPrn ("diff %d", (int) (curr -> mhash - next -> mhash));

		if (curr -> mhash > next -> mhash) {
			// CudaDbgPrn ("curr -> mhash > next -> mhash - true");
			return true;
		}
		if (curr -> mhash == next -> mhash) {
			if (curr -> str [0] > next -> str [0]) {
				// CudaDbgPrn ("curr -> str [0] > next -> str [0] - true");
				return true;
			}
			else if ((curr -> str [0] == next -> str [0]) && (curr -> str [1] > next -> str [1])) {
				// CudaDbgPrn ("(curr -> str [0] == next -> str [0]) && (curr -> str [1] > next -> str [1]) - true");
				return true;
			}
		}
	}

	// CudaDbgPrn ("False: mhash: %u/%u: curr: %.2s next %.2s", (unsigned) curr -> mhash, (unsigned) next -> mhash, curr -> str, next -> str);
	return false;
}

bool __device__ __host__ isEqual (const mhash_vals *curr, const mhash_vals *next)
{
	if (curr -> mhash && curr -> mhash == next -> mhash) {
		// CudaDbgPrn ("curr/next match mhash: %u/%u: st:%.2s/%.2s", (unsigned) curr -> mhash, (unsigned) next -> mhash, curr -> str, next -> str);

		if ((curr -> str [0] == next -> str [0]) && (curr -> str [1] == next -> str [1])) {
			// CudaDbgPrn ("True: curr/next mhash: %u/%u st: %.2s/%.2s", (unsigned) curr -> mhash, (unsigned) next -> mhash, curr -> str, next -> str);
			return true;
		}
	}

	// CudaDbgPrn ("False: mhash: %u/%u: curr: %.2s next %.2s", (unsigned) curr -> mhash, (unsigned) next -> mhash, curr -> str, next -> str);
	return false;
}

/*********************
 * Usage:
 * Declare compress_sem globally: __device__ volatile int compress_sem = 0;
 *
 * acquire_semaphore (&compress_sem);
 * // critical section
 * release_semaphore (&compress_sem);
 *********************/
__device__ void acquire_semaphore (volatile int *lock)
{
	__syncthreads ();
	while (0 != atomicCAS ((int *) lock, 0, 1));
	__syncthreads ();
}

__device__ void release_semaphore (volatile int *lock)
{
	__syncthreads ();
	*lock = 0;
	__threadfence ();
}
