/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * sort_merge.cu
 ******************************************************/

#include <cuda_milib.h>
#include <milib.h>

/*************  Exported Functions  **************/
void milib_gpu_sort_merge_histo_wds (all_bufs *ab, bool stop_words);

/*************  External Functions  **************/
extern token_tracking *create_token_tracking (gpu_config *gc, const unsigned int chunk_size);
extern void update_h_ttrack (gpu_calc *gb, token_tracking *h_ttrack);
extern void memory_compaction (all_bufs *ab);
extern void apply_stop_words (token_tracking *hs, all_bufs *ab);

extern void set_to_zero (unsigned int *d_val);
extern unsigned int get_d_val (unsigned int *d_val);


extern bool __device__ __host__ isGreater (const mhash_vals *curr, const mhash_vals *next);
extern bool __device__ __host__ isEqual (const mhash_vals *curr, const mhash_vals *next);
extern __device__ void cuda_swap_mem (tok_mhash *d_mem1, tok_mhash *d_mem2);

// test functions
extern void print_token_tracking (gpu_calc *gb, token_tracking *hst);
extern void print_token_from_device (gpu_calc *gb, token_tracking *tt);
extern void print_scratchpad_and_chunks (gpu_calc *gb, token_tracking *scr, token_tracking *src);

/*************  structures local to this file **************/
#define TRUE			1
#define FALSE			0

/***
 * struct for sort_one_token_block
 * one block has 1-n chunks.
 * tt has one chunk, scratchpad has n chunks
 * one block has n_warps - different for tt & for scratchpad: calculate for each
 * n_chunks impacts allocation of d_data memory
 * n_warps impacts calculation of grid
 ***/
typedef struct MY_ALIGN (16) data_sort_otc {
	unsigned int	n_chunks;		// total shmem chunks in chunk of tt
	unsigned int	n_warps;		// total warps in chunk of tt
	bool			is_stop_word;	// is this sort for stop_words or main sort

	volatile unsigned int	*d_is_K_Sorted;		// K_Sort_Merge
	volatile unsigned int	*d_is_core_sorted;	// K_core_sort
} data_sort_otc;

__device__ volatile unsigned int	d_Srt [1] = {FALSE};	// high level sort

/*************  functions  **************/
static unsigned int get_sort_status (volatile unsigned int *d_Srtd);
static unsigned int get_sort_status (volatile unsigned int *d_Srtd)
{
	unsigned int h_srt = FALSE;
	gpuErrChk (cudaMemcpy (&h_srt, (void *) d_Srtd, sizeof (unsigned int), cudaMemcpyDeviceToHost));

	return h_srt;
}

static void set_sort_true (volatile unsigned int *d_Srtd);
static void set_sort_true (volatile unsigned int *d_Srtd)
{
	unsigned int h_srt = TRUE;
	gpuErrChk (cudaMemcpy ((void *) d_Srtd, &h_srt, sizeof (unsigned int), cudaMemcpyHostToDevice));
}

static void set_sort_false (volatile unsigned int *d_Srtd);
static void set_sort_false (volatile unsigned int *d_Srtd)
{
	unsigned int h_srt = FALSE;
	gpuErrChk (cudaMemcpy ((void *) d_Srtd, &h_srt, sizeof (unsigned int), cudaMemcpyHostToDevice));
}

/*************  start main sort Functions  **************/
/***
 * ct Assumption: ALL of curr [th] - *tok is on ct.
 * zero out curr *tok no problem. zeroing out *tok contents could have issue
 * DON'T zero *tok contents EXCEPT in MERGE.
 * TBD: RS_DEBUG
 * The three alternatives are:
 * 1. chunk_size = 1/8th of maxThreads & scratchpad = maxThreads
 * 2. chunk_size = half of maxThreads & scratchpad = maxThreads * 4
 * 3. chunk_size = maxThreads & scratchpad = tot_mem / 8
 * We choose alternative 1 for implementation.  We should check performance for
 * alternatives 2 & 3.
 * We should also explore other sort mechanisms.
 ***/
static void __device__ K_core_sort (tok_mhash *ct, const unsigned int gTh, const data_sort_otc *k_sotc, const unsigned int chunk_size, unsigned int *curr_num_free);
static inline __device__ action isAction (tok_mhash *curr, tok_mhash *next);
static inline __device__ bool willSort (const unsigned int half, const unsigned int chunk, const unsigned int tot_chunks);

static void __global__ K_Sort_Merge (token **curr, unsigned int *curr_num_free, const data_sort_otc *k_sotc, const unsigned int chunk_size, uint32_t *f_hash);
static void __global__ K_Sort_Merge (token **curr, unsigned int *curr_num_free, const data_sort_otc *k_sotc, const unsigned int chunk_size, uint32_t *f_hash)
{
	extern __shared__ tok_mhash ct [];	// to swap in local mem
	const unsigned gTh	= blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= chunk_size) return;

	// unsigned int i = (unsigned int) 0;	// RS_DEBUG
	while ((unsigned int) FALSE == *k_sotc -> d_is_K_Sorted) {
		atomicExch ((unsigned int *) k_sotc -> d_is_K_Sorted, (unsigned int) TRUE);

		/***
		CudaDbgPrn ("th %u:loop %u:shmem (max gTh) %u n_chunks %i chunk_size %u, is_stop_word %u", (unsigned) gTh, (unsigned) ++i, (unsigned) chunk_size, (unsigned) k_sotc -> n_chunks, (unsigned) chunk_size, (unsigned) k_sotc -> is_stop_word);
		***/

		for (unsigned int chunk = 0; chunk < k_sotc -> n_chunks; chunk++) {
			unsigned int  blk_loc = 0;

#pragma unroll
			for (unsigned int isHalf = 0; isHalf < 2; isHalf++) {	// half chunk
				if (true == willSort (isHalf, chunk, k_sotc -> n_chunks)) {
					cuda_MemSet (&ct [gTh], (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
					blk_loc = gTh + chunk * chunk_size +
						isHalf * chunk_size / 2;	// curr loc in block

					if (curr [blk_loc]) {
						ct [gTh].tok = (token *) curr [blk_loc];
						cuda_MemCpy (&ct [gTh].mhv, &curr [blk_loc] -> mhv, sizeof (mhash_vals));
					}
					atomicExch ((unsigned int *) k_sotc -> d_is_core_sorted, (unsigned int) FALSE);	// set FALSE before shmem sort
					__syncthreads ();
					/***
					if (ct [gTh].mhv.mhash) {
							CudaDbgPrn ("Before K_core_sort :gTh:%u free:%u n_chunks:%u is_stop_word %u|curr[gTh:%u] %p ct[gTh:%u]: tok %p; mhash %u %.2s", (unsigned) gTh, (unsigned) *curr_num_free, (unsigned) k_sotc -> n_chunks, (unsigned) k_sotc -> is_stop_word, curr [gTh], (unsigned) gTh, ct [gTh].tok, (unsigned) ct [gTh].tok -> mhv.mhash, ct [gTh].tok -> mhv.str);
					}
					else {
						CudaDbgPrn ("Before K_core_sort :gTh:%u free:%u n_chunks:%u is_stop_word %u|curr[gTh:%u] %p ct[gTh:%u]: tok (null)", (unsigned) gTh, (unsigned) *curr_num_free, (unsigned) k_sotc -> n_chunks, (unsigned) k_sotc -> is_stop_word, curr [gTh], (unsigned) gTh);
					}
					***/

					K_core_sort (ct, gTh, k_sotc, chunk_size, curr_num_free);

					/***
					if (ct [gTh].mhv.mhash) {
						CudaDbgPrn ("After :gTh:%u free:%u n_chunks:%u is_stop_word %u|ct[gTh:%u]: mhash %u %.2s", (unsigned) gTh, (unsigned) *curr_num_free, (unsigned) k_sotc -> n_chunks, (unsigned) k_sotc -> is_stop_word, (unsigned) gTh, (unsigned) ct [gTh].tok -> mhv.mhash, ct [gTh].tok -> mhv.str);
					}
					else {CudaDbgPrn ("After :gTh:%u free:%u n_chunks:%u is_stop_word %u|ct[gTh:%u]: tok (null)", (unsigned) gTh, (unsigned) *curr_num_free, (unsigned) k_sotc -> n_chunks, (unsigned) k_sotc -> is_stop_word, (unsigned) gTh);}
					***/

					curr [blk_loc] = (token *) ct [gTh].tok;
					// KER_RETURN;
				}
				__syncthreads ();
			}
		}
	}

	if (0 == gTh && f_hash) {
		*f_hash = (uint32_t) 0;
		if (ct [0].tok) {*f_hash = (uint32_t) ct [0].mhv.mhash;}
	}

	/***
	if (curr [gTh]) {
		CudaDbgPrn ("final th%u free %u|%.15s hsh %u nm %u", (unsigned) gTh, (unsigned) *curr_num_free, curr [gTh] -> wd.str, (unsigned) curr [gTh] -> mhv.mhash, (unsigned) curr [gTh] -> wd.num);
	}
	else {CudaDbgPrn ("final curr [gTh: %u] (null)", (unsigned) gTh);}
	if (0 == gTh) {CudaDbgPrn ("i %u", (unsigned) ++i);}
	***/
}

static inline __device__ bool willSort (const unsigned int half, const unsigned int chunk, const unsigned int tot_chunks)
{
	if (1 == half && 1 == tot_chunks) {
		return false;
	}
	if (1 < tot_chunks && (chunk + 1) >= tot_chunks) {
		return false;
	}

	return true;
}

static void __device__ K_core_sort (tok_mhash *ct, const unsigned int gTh, const data_sort_otc *k_sotc, const unsigned int chunk_size, unsigned int *curr_num_free)
{
	register unsigned int curr_loc, next_loc;
	// CudaDbgPrn ("shmem %u free %u is_stop_word %u|ct[%u]: mhash %u %.2s", (unsigned) shmem, (unsigned) *curr_num_free, (unsigned) is_stop_word, (unsigned) gTh, (unsigned) ct [gTh].mhv.mhash, ct [gTh].mhv.str);

	while ((unsigned int) FALSE == *k_sotc -> d_is_core_sorted) {
		atomicExch ((unsigned int *) k_sotc -> d_is_core_sorted, (unsigned int) TRUE);

#pragma unroll
		for (unsigned int i = 0; i < 2; i++) {	// even, odd
			curr_loc = (unsigned int) (gTh << 1) + i;
			next_loc = (unsigned int) curr_loc + 1;

			action act;
			if (chunk_size > next_loc) {
				act = isAction ((tok_mhash *) &ct [curr_loc], (tok_mhash *) &ct [next_loc]);

				/***
				if (NONE != act) {
					CudaDbgPrn ("NOT NONE th %u curr %u next %u act %u", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc, (unsigned) act);
				}
				else {
					CudaDbgPrn ("NONE th %u c:%u n:%u act %i free%u", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc, (unsigned) act, (unsigned) *curr_num_free);
				} ***/

				if (NONE != act && !(k_sotc -> is_stop_word && MERGE == act)) {
					*k_sotc -> d_is_core_sorted = (unsigned int) FALSE;
					*k_sotc -> d_is_K_Sorted = (unsigned int) FALSE;
					*d_Srt = (unsigned int) FALSE;
					__threadfence ();

					if (MOVE == act) {
						ct [next_loc].tok = (token *) ct [curr_loc].tok;
						cuda_MemCpy (&ct [next_loc].mhv, &ct [curr_loc].mhv, sizeof (mhash_vals));
						cuda_MemSet (&ct [curr_loc], (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));

						/***
						if (ct[curr_loc].tok) {CudaDbgPrn ("NULL SWAPped curr: ct [%u].tok %p", (unsigned) curr_loc, ct [curr_loc].tok);}
						if (ct[next_loc].tok) {CudaDbgPrn ("NULL SWAPped next: ct [%u].tok mhash %u %.2s", (unsigned) next_loc, (unsigned) ct [next_loc].tok -> mhv.mhash, ct [next_loc].tok -> mhv.str);}
						***/
					}
					else if (SWAP == act) {
						cuda_swap_mem (&ct [curr_loc], &ct [next_loc]);

						/***
						if (ct[curr_loc].tok) {CudaDbgPrn ("mem SWAPped curr: ct [%u].tok %p mhash %u %.2s", (unsigned) curr_loc, ct [curr_loc].tok, (unsigned) ct [curr_loc].tok -> mhv.mhash, ct [curr_loc].tok -> mhv.str);}
						if (ct[next_loc].tok) {CudaDbgPrn ("mem SWAPped next: ct [%u].tok mhash %u %.2s", (unsigned) next_loc, (unsigned) ct [next_loc].tok -> mhv.mhash, ct [next_loc].tok -> mhv.str);}
						***/
					}
					else if (MERGE == act && false == k_sotc -> is_stop_word) {
						// CudaDbgPrn ("%u:Merge c:%u/n:%u", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc);
						ct [next_loc].tok -> wd.num += ct [curr_loc].tok -> wd.num;
						cuda_MemSet (&ct [curr_loc], (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
						atomicAdd (curr_num_free, (unsigned int) 1);
					}

					/***
					CudaDbgPrn ("after: c/n:%u/%u ptr:%p/%p act %u", (unsigned) curr_loc, (unsigned) next_loc, ct [curr_loc].tok, ct [next_loc].tok, (unsigned) act);

					if (NONE == act) {
						CudaDbgPrn ("after: free%u shmem %u c%u/n%u act NONE", (unsigned) *curr_num_free, (unsigned) shmem, (unsigned) curr_loc, (unsigned) next_loc);
					}
					else if (ct [curr_loc].tok && 0 != ct [curr_loc].mhv.mhash) {
						CudaDbgPrn ("after: curr: ct [%u].tok mhash %u", (unsigned) curr_loc, (unsigned) ct [curr_loc].tok -> mhv.mhash);
					}
					else {CudaDbgPrn ("after_act curr: ct [%u].tok (null)", (unsigned) curr_loc);}
					if (NONE != act && ct [next_loc].tok && 0 != ct [next_loc].mhv.mhash) {
						assert (NULL != ct [next_loc].tok);
						assert (0 != ct [next_loc].mhv.mhash);
						CudaDbgPrn ("after_act %u next: ct [%u].tok mhash %u %.2s", (unsigned) act, (unsigned) next_loc, (unsigned) ct [next_loc].tok -> mhv.mhash, ct [next_loc].tok -> mhv.str);
					}
					else {CudaDbgPrn ("after_act %u next: ct [%u].tok (null)", (unsigned) act, (unsigned) next_loc);}
					***/
				}
			}
			// CudaDbgPrn ("inside for: KSrt %d", (int) *d_is_core_sorted);
			__threadfence ();
			__syncthreads ();
		}
	}

	/***
	if (ct [gTh].tok && ct [gTh].mhv.mhash) {
		CudaDbgPrn ("end: th%u/c%u/n%u free%u KSrt %d|%p hsh %u", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc, (unsigned) *curr_num_free, (int) *d_is_core_sorted, ct [gTh].tok, (unsigned) ct [gTh].mhv.mhash);
		// CudaDbgPrn ("end: th%u/c%u/n%u free%u KSrt %d|%.15s hsh %u nm %u", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc, (unsigned) *curr_num_free, (int) *d_is_core_sorted, ct [gTh].tok -> wd.str, (unsigned) ct [gTh].mhv.mhash, (unsigned) ct [gTh].tok -> wd.num);
	}
	else {
		CudaDbgPrn ("end: th%u/c%u/n%u free%u KSrt %d|ct [%u].tok (null)", (unsigned) gTh, (unsigned) curr_loc, (unsigned) next_loc, (unsigned) *curr_num_free, (int) *d_is_core_sorted, gTh);
	}
	***/
}

static inline __device__ action isAction (tok_mhash *curr, tok_mhash *next)
{
	// CudaDbgPrn ("curr -> tok %p next -> tok %p", curr -> tok, next -> tok);

	if (!curr -> tok) {return NONE;}
	if (next -> tok) {
		if (true == isGreater (&curr -> mhv, &next -> mhv)) {
			// CudaDbgPrn ("curr %u|next %u greater - SWAP", (unsigned) curr -> mhv	.mhash, (unsigned) next  -> mhv.mhash);
			return SWAP;
		}

		if (true == isEqual (&curr -> mhv, &next -> mhv)) {
			// CudaDbgPrn ("curr %u|next %u equal - MERGE", (unsigned) curr -> mhv	.mhash, (unsigned) next  -> mhv.mhash);
			return MERGE;
		}
	}
	else {
		// CudaDbgPrn ("curr %u|next %u MOVE", (unsigned) curr -> mhv	.mhash, (unsigned) next  -> mhv.mhash);
		return MOVE;
	}

	// CudaDbgPrn ("curr %u|next %u return NONE", (unsigned) curr -> mhv	.mhash, (unsigned) next  -> mhv.mhash);
	return NONE;
}

static void milib_sort_across_block (gpu_calc *gb, token_tracking *tt, data_sort_otc *d_sotc);
static unsigned int sort_one_token_block (gpu_calc *gb, token_tracking *tt, data_sort_otc *d_sotc);
static unsigned int sort_one_token_block (gpu_calc *gb, token_tracking *tt, data_sort_otc *d_sotc)
{
	milib_sort_across_block (gb, tt, d_sotc);

	static uint32_t		*d_first_hash = NULL;
	if (!d_first_hash) {
		gpuErrChk (cudaMalloc (&d_first_hash, sizeof (uint32_t)));
	}

	gpuErrChk (cudaMemcpy (&(tt -> h_num_free), tt -> d_num_free, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	dim3 block = gb -> block;
	dim3 grid ((tt -> chunk_size + block.x - 1) / block.x);
	data_sort_otc h_sotc;
	gpuErrChk (cudaMemcpy (&h_sotc, d_sotc, sizeof (data_sort_otc), cudaMemcpyDeviceToHost));

	set_sort_false (h_sotc.d_is_K_Sorted);
	if (!h_sotc.is_stop_word) {
		K_Sort_Merge <<<grid, block, gb -> block_size * sizeof (tok_mhash)>>> (tt -> d_data, tt -> d_num_free, d_sotc, gb -> block_size, d_first_hash);
		gpuErrChk (cudaMemcpy (&(tt -> first_hash), d_first_hash, sizeof (uint32_t), cudaMemcpyDeviceToHost));
	}
	else {
		K_Sort_Merge <<<grid, block, gb -> block_size * sizeof (tok_mhash)>>> (tt -> d_data, tt -> d_num_free, d_sotc, gb -> block_size, NULL);
		tt -> first_hash = 0;
	}
	CUDA_GLOBAL_CHECK;

	return tt -> h_num_free;
}

static void __global__ K_Sort_across_half_block (token **block, const data_sort_otc *d_sotc, const unsigned int c_size, const unsigned int mid, unsigned int *num_free);
static void __global__ K_Sort_across_half_block (token **block, const data_sort_otc *d_sotc, const unsigned int c_size, const unsigned int mid, unsigned int *num_free)
{
	const unsigned int gTh = (blockIdx.x * blockDim.x + threadIdx.x);
	// Sort across first half & 2nd half

	const unsigned int c_th = gTh;			// loc of current tok
	const unsigned int n_th = gTh + mid;	// loc of next tok
	if (c_size <= n_th) {return;}
	// CudaDbgPrn ("c_th %u n_th %u", (unsigned) c_th, (unsigned) n_th);

	tok_mhash curr_tok, next_tok;
	cuda_MemSet (&curr_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
	cuda_MemSet (&next_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));

	if (block [c_th]) {
		cuda_MemCpy (&(curr_tok.mhv), &(block [c_th] -> mhv), sizeof (mhash_vals));	// global to register
		curr_tok.tok = (token *) block [c_th];
	}
	else {return;}			// curr == 0, don't swap
	if (block [n_th]) {
		cuda_MemCpy (&(next_tok.mhv), &(block [n_th] -> mhv), sizeof (mhash_vals));	// copy global to register
		next_tok.tok = (token *) block [n_th];
	}

	action act = isAction ((tok_mhash *) &curr_tok, (tok_mhash *) &next_tok);
	if (NONE == act) {return;}
	if (SWAP == act) {
		if (NULL == next_tok.tok) {
			next_tok.tok = (token *) curr_tok.tok;
			cuda_MemCpy (&next_tok.mhv, &curr_tok.mhv, sizeof (mhash_vals));
			cuda_MemSet (&curr_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
		}
		else {
			cuda_swap_mem (&curr_tok, &next_tok);
		}

		block [c_th] = (token *) curr_tok.tok;		// register to global
		block [n_th] = (token *) next_tok.tok;
		// CudaDbgPrn ("th:%u: c:%u/n:%u Swapped", (unsigned) gTh, (unsigned) c_th, (unsigned) n_th);
	}
	else if (MERGE == act && false == d_sotc -> is_stop_word) {
		block [n_th] -> wd.num += block [c_th] -> wd.num;
		block [c_th] = (token *) NULL;
		atomicAdd (num_free, (unsigned int) 1);
		// CudaDbgPrn ("th:%u: c:%u/n:%u Merged", (unsigned) gTh, (unsigned) c_th, (unsigned) n_th);
	}
	atomicExch ((unsigned int *) d_Srt, (unsigned int) FALSE);

	/***
	CudaDbgPrn ("gTh %u act %s, c.mhash %u", gTh, (0 == act)?"SWAP":((1 == act)?"MERGE":"NONE"), (unsigned) curr_tok.mhv.mhash);

	if (curr_tok.tok && next_tok.tok) {
		CudaDbgPrn ("Quarter: gTh %u c: str %s hash %u  num %u\n\tn: str %s hash %u  num %u", (unsigned) gTh, curr_tok.mhv.str, (unsigned) curr_tok.mhv.mhash, (unsigned) curr_tok.tok -> wd.num, next_tok.mhv.str, (unsigned) next_tok.mhv.mhash, (unsigned) next_tok.tok -> wd.num);
	}
	***/
}

/***
 * TBD: RS_DEBUG will implement in the future
 * need to think this through properly
 * TBD:This implementation is incomplete.
static void __global__ K_Sort_across_adjacent_warps (token **block, const data_sort_otc *d_sotc, unsigned int *num_free);
static void __global__ K_Sort_across_adjacent_warps (token **block, const data_sort_otc *d_sotc, unsigned int *num_free)
{
	const unsigned int gTh = (blockIdx.x * blockDim.x + threadIdx.x);
	// unsigned int i = (unsigned int) 0;	// RS_DEBUG
	atomicExch ((unsigned int *) d_sotc -> d_is_K_Sorted, (unsigned int) FALSE);
	__syncthreads ();

	while ((unsigned int) FALSE == *d_sotc -> d_is_K_Sorted) {
		atomicExch ((unsigned int *) d_sotc -> d_is_K_Sorted, (unsigned int) TRUE);

		// CudaDbgPrn ("th %u:loop %u:shmem (max gTh) %u n_chunks %i chunk_size %u, is_stop_word %u", (unsigned) gTh, (unsigned) ++i, (unsigned) d_sotc -> chunk_size, (unsigned) d_sotc -> n_chunks, (unsigned) d_sotc -> chunk_size, (unsigned) d_sotc -> is_stop_word);

#pragma unroll
		for (unsigned int i = 0; i < 2; i++) {			// 1st, then 2nd set
			unsigned int c_th, n_th;
			c_th = gTh + i;								// loc of current tok
			n_th = gTh + d_sotc -> warp_size + i;		// loc of next tok
			if (d_sotc -> block_size <= n_th) continue;

			tok_mhash curr_tok, next_tok;
			cuda_MemSet (&curr_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
			cuda_MemSet (&next_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));

			if (block [c_th]) {
				cuda_MemCpy (&(curr_tok.mhv), &(block [c_th] -> mhv), sizeof (mhash_vals));						// global to register
				curr_tok.tok = block [c_th];
			}
			if (block [n_th]) {
				cuda_MemCpy (&(next_tok.mhv), &(block [n_th] -> mhv), sizeof (mhash_vals));						// global to register
				next_tok.tok = block [n_th];
			}

			action act = isAction (&curr_tok, &next_tok);
			if (SWAP == act) {
				if (NULL == next_tok.tok) {
					next_tok.tok = (token *) curr_tok.tok;
					cuda_MemCpy (&next_tok.mhv, &curr_tok.mhv, sizeof (mhash_vals));
					cuda_MemSet (&curr_tok, (uint8_t) 0x00, (uint32_t) (sizeof (tok_mhash)));
				}
				else {cuda_swap_mem (&curr_tok, &next_tok);}

				atomicExch ((unsigned int *) d_sotc -> d_is_K_Sorted, (unsigned int) FALSE);
				block [c_th] = (token *) curr_tok.tok;	// register to global
				block [n_th] = (token *) next_tok.tok;
				// CudaDbgPrn ("%u: Quarter: Swapped", (unsigned) gTh);
			}
			else if (MERGE == act && false == d_sotc -> is_stop_word) {
				block [n_th] -> wd.num += block [c_th] -> wd.num;
				block [c_th] = (token *) NULL;
				atomicAdd (num_free, (unsigned int) 1);
				atomicExch ((unsigned int *) d_sotc -> d_is_K_Sorted, (unsigned int) FALSE);
			}

			CudaDbgPrn ("gTh %u act %s, c.mhash %u", gTh, (0 == act)?"SWAP":((1 == act)?"MERGE":"NONE"), (unsigned) c -> mhv.mhash);

			if (n -> mhv.mhash) {
				CudaDbgPrn ("Quarter: gTh %u c: str %s hash %u  num %u\n\tn: str %s hash %u  num %u", (unsigned) gTh, curr -> str, (unsigned) curr -> mhv.mhash, (unsigned) curr -> num, next -> str, (unsigned) next -> mhv.mhash, (unsigned) next -> num);
			}

			__syncthreads ();
		}
	}
}
***/

static void milib_sort_across_block (gpu_calc *gb, token_tracking *tt, data_sort_otc *d_sotc)
{
	if (gb -> warp_size < tt -> chunk_size) {
		dim3 block (gb -> warp_size);
		dim3 grid ((tt -> chunk_size + block.x -1) / block.x);

		K_Sort_across_half_block <<<grid, block>>> (tt -> d_data, d_sotc, tt -> chunk_size, tt -> chunk_size / 2, tt -> d_num_free);
		CUDA_GLOBAL_CHECK;

		/***
		 * TBD: RS_DEBUG will implement in the future
		K_Sort_across_adjacent_warps <<<grid, block>>> (tt -> d_data, d_sotc, tt -> d_num_free);
		CUDA_GLOBAL_CHECK;
		***/
	}

	// Dbg ("======== milib_sort_across_block done ========");
}

/*************  end main sort Functions  **************/

static bool sort_merge_histo_cross (all_bufs *ab, token_tracking **curr, data_sort_otc *d_sotc);
static token_tracking *sort_token_chunks (token_tracking *h_ttrack);

void milib_gpu_sort_merge_histo_wds (all_bufs *ab, bool is_stop_words)
{
	unsigned int wds;
	if (is_stop_words) wds = ab -> h_Swds;
	else {
		gpuErrChk (cudaMemcpy (&wds, ab -> d_wds, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	}

	static unsigned int *loc_d_is_K_Sorted = NULL,
					*loc_d_is_core_sorted = NULL;
	if (!loc_d_is_K_Sorted) {gpuErrChk (cudaMalloc (&loc_d_is_K_Sorted, sizeof (unsigned int)));}
	if (!loc_d_is_core_sorted) {gpuErrChk (cudaMalloc (&loc_d_is_core_sorted, sizeof (unsigned int)));}

	data_sort_otc hd_sotc;
	hd_sotc.n_chunks = 1;		// default is ALWAYS one chunk
	hd_sotc.n_warps = ab -> def_gpu_calc.block_size / ab -> def_gpu_calc.warp_size;
	hd_sotc.is_stop_word = is_stop_words;

	hd_sotc.d_is_K_Sorted = loc_d_is_K_Sorted;
	hd_sotc.d_is_core_sorted = loc_d_is_core_sorted;

	static data_sort_otc *d_sotc = NULL;
	if (!d_sotc) {gpuErrChk (cudaMalloc (&d_sotc, sizeof (data_sort_otc)));}
	gpuErrChk (cudaMemcpy (d_sotc, &hd_sotc, sizeof (data_sort_otc), cudaMemcpyHostToDevice));	// sotc is now in GPU

	token_tracking *tmp;
	for (tmp = ab -> h_ttrack; tmp; tmp = tmp -> next) {
		// apply stop_words only when called from main, not while processing
		// stop_words itself
		if (ab -> IsS_wd && !is_stop_words) {
			apply_stop_words (tmp, ab);
			// Dbg ("------ after apply_stop_words: ");
		}

		tmp -> h_num_free = sort_one_token_block (&ab -> def_gpu_calc, tmp, d_sotc);	// stop_words:no merge

		if (!is_stop_words) {
			update_h_ttrack (&ab -> def_gpu_calc, tmp);
		}
	}

	/***
	Dbg ("\n\n------ printing histo BEFORE sort_merge cross -------");
	token_tracking *tst = ab -> h_ttrack;	// RS_DEBUG
	while (tst && tst -> prev) tst = tst -> prev;
	for (; tst; tst = tst -> next) PRINT_HISTO (&ab -> def_gpu_calc, tst);
	***/

	set_sort_false (hd_sotc.d_is_K_Sorted);
	bool toCompact = false;	// shall we run memory_compaction?
	while (false == get_sort_status (hd_sotc.d_is_K_Sorted)) {
		set_sort_true (hd_sotc.d_is_K_Sorted);	// sorting across multiple funcs.

		tmp = sort_token_chunks (ab -> h_ttrack);
		while (ab -> h_ttrack -> prev) ab -> h_ttrack = ab -> h_ttrack -> prev;
		if (!tmp) tmp = ab -> h_ttrack;
		sort_merge_histo_cross (ab, &tmp, d_sotc);

		toCompact = false;
		if (tmp -> chunk_size == tmp -> h_num_free) toCompact = true;

		/***
		Dbg ("\n\n------ printing histo BEFORE memory_compaction -------");
		tst = ab -> h_ttrack;	// RS_DEBUG
		while (tst && tst -> prev) tst = tst -> prev;
		for (; tst; tst = tst -> next) PRINT_HISTO (&ab -> def_gpu_calc, tst);
		***/

		if (toCompact && false == is_stop_words) {
			memory_compaction (ab);
			toCompact = false;
		}

		/***
		Dbg ("\n\n------ printing histo AFTER memory_compaction -------");
		tst = ab -> h_ttrack;	// RS_DEBUG
		while (tst && tst -> prev) tst = tst -> prev;
		for (; tst; tst = tst -> next) PRINT_HISTO (&ab -> def_gpu_calc, tst);
		***/
	}
}

/************* start sort_merge_histo_cross functions ***************/
// Assumption: After moving chunk left, that part of scratchpad is zeroed out
// & scratchpad -> h_num_free is increased accordingly
// in actuality, the scratchpad is NOT zeroed.
static void mv_one_chunk_left (token_tracking *scratchpad, const size_t chunks, const size_t chunk_size);
static void mv_one_chunk_left (token_tracking *scratchpad, const size_t chunks, const size_t chunk_size)
{
	gpuErrChk (cudaMemcpy (scratchpad -> d_data, scratchpad -> d_data + chunk_size, sizeof (token **) * chunk_size * (chunks - 1), cudaMemcpyDeviceToDevice));

	scratchpad -> h_num_free += chunk_size;

	gpuErrChk (cudaMemcpy (scratchpad -> d_num_free, &scratchpad -> h_num_free, sizeof (unsigned int), cudaMemcpyHostToDevice));
}

/*********** start sort_token_chunks functions *****************/
static token_tracking *find_first_tt_head_with_data (token_tracking *head);
static token_tracking *swap_token_chunk (token_tracking *curr, token_tracking *next);

static token_tracking *sort_token_chunks (token_tracking *h_ttrack)
{
	if (!h_ttrack -> prev && !h_ttrack -> next) return h_ttrack;	// don't sort

	bool isSorted = false;
	// Assumption: since not many chunks, simple swap sorts list
	while (false == isSorted) {
		isSorted = true;
		token_tracking *tmp, *nxt = h_ttrack -> next;
		for (tmp = find_first_tt_head_with_data (h_ttrack); tmp && nxt;
			 tmp = tmp -> next) {
			nxt = tmp -> next;
			if (nxt) {
				if (!tmp -> first_hash) {
					if (tmp -> h_num_free < nxt -> h_num_free) {
						tmp = swap_token_chunk (tmp, nxt);
						isSorted = false;	// sorting within this func
					}
				}
				else {
					if (tmp -> first_hash > nxt -> first_hash) {
						tmp = swap_token_chunk (tmp, nxt);
						isSorted = false;	// sorting within this func
					}
				}
			}

			h_ttrack = find_first_tt_head_with_data (h_ttrack);
		 }
	}

	// while (h_ttrack -> prev) h_ttrack = h_ttrack -> prev; for testing
	return h_ttrack;			// only tokens with contents
}

// These are the index values for links
#define PREV						0
#define CURR						1
#define NEXT						2
#define FUTR						3
static token_tracking *swap_token_chunk (token_tracking *curr, token_tracking *next)
{
	token_tracking *links [4];		// all the links for swapping

	links [PREV] = curr -> prev;	// previous
	links [CURR] = curr;			// current
	links [NEXT] = next;			// next
	links [FUTR] = next -> next;	// future

	links [CURR] -> prev = links [NEXT];
	links [CURR] -> next = links [FUTR];
	links [NEXT] -> prev = links [PREV];
	links [NEXT] -> next = links [CURR];
	if (links [PREV]) links [PREV] -> next = links [NEXT];
	if (links [FUTR]) links [FUTR] -> prev = links [CURR];

	return next;
}

static token_tracking *find_first_tt_head_with_data (token_tracking *head)
{
	while (head -> prev) head = head -> prev;

	while (head) {
		if (head -> chunk_size == head -> h_num_free) head = head -> next;
		else break;
	}

	return head;
}
/*********** end sort_token_chunks functions *****************/

/************* start src_to_scratchpad functions ***************/
static void copy_src_to_scratchpad (token_tracking *scratchpad, const unsigned int start, token_tracking *src);
static void copy_src_to_scratchpad (token_tracking *scratchpad, const unsigned int start, token_tracking *src)
{
	gpuErrChk (cudaMemcpy (scratchpad -> d_data + start * src -> chunk_size, src -> d_data, sizeof (token **) * src -> chunk_size, cudaMemcpyDeviceToDevice));

	if (scratchpad -> h_num_free >= src -> chunk_size) {
		scratchpad -> h_num_free -= (src -> chunk_size - src -> h_num_free);
		gpuErrChk (cudaMemcpy (scratchpad -> d_num_free, &scratchpad -> h_num_free, sizeof (unsigned int), cudaMemcpyHostToDevice));
		src -> h_num_free = 0;
	}
	else {
		Dbg ("Error scratchpad -> h_num_free < src -> chunk_size");
		assert (scratchpad -> h_num_free >= src -> chunk_size);
		exit (1);
	}
}

static token_tracking *copy_multi_src_to_scratchpad (token_tracking *scratchpad, token_tracking *src, const unsigned int num);
static token_tracking *copy_multi_src_to_scratchpad (token_tracking *scratchpad, token_tracking *src, const unsigned int num)
{
	for (unsigned i = 0; i < num && src; i++, src = src -> next) {
		copy_src_to_scratchpad (scratchpad, i, src);
	}

	return src;
}
/************* src_to_scratchpad functions ***************/
/************* scratchpad_to_tgt functions ***************/
// returns the updated tgt.
// Assumption: After copying a chunk to tt, that part of scratchpad is zeroed out
// & scratchpad -> h_num_free is increased accordingly. tgt -> h_num_free is set
// in actuality, the scratchpad is NOT zeroed.
static bool copy_scratchpad_to_tgt (gpu_calc *gb, token_tracking *tgt, token_tracking *scratchpad, unsigned int loc = 0);
static bool copy_scratchpad_to_tgt (gpu_calc *gb, token_tracking *tgt, token_tracking *scratchpad, unsigned int loc)
{
	bool ret;
	// tgt -> chunk_size always <= scratchpad -> h_num_free
	if (scratchpad -> h_num_free >= tgt -> chunk_size) {
		scratchpad -> h_num_free -= tgt -> chunk_size;
		tgt -> h_num_free = tgt -> chunk_size;

		ret = false;
	}
	else if (scratchpad -> h_num_free) {	// scratchpad->free < tgt->size
		gpuErrChk (cudaMemcpy (tgt -> d_data, scratchpad -> d_data + (loc * tgt -> chunk_size), sizeof (token **) * tgt -> chunk_size, cudaMemcpyDeviceToDevice));
		tgt -> h_num_free = scratchpad -> h_num_free;
		scratchpad -> h_num_free = 0;
		ret = true;
	}
	else {		// scratchpad -> h_num_free ==0, is full of data
		gpuErrChk (cudaMemcpy (tgt -> d_data, scratchpad -> d_data + (loc * tgt -> chunk_size), sizeof (token **) * tgt -> chunk_size, cudaMemcpyDeviceToDevice));
		tgt -> h_num_free = 0;
		ret = true;
	}

	gpuErrChk (cudaMemcpy (scratchpad -> d_num_free, &scratchpad -> h_num_free, sizeof (unsigned int), cudaMemcpyHostToDevice));
	gpuErrChk (cudaMemcpy (tgt -> d_num_free, &tgt -> h_num_free, sizeof (unsigned int), cudaMemcpyHostToDevice));

	if (true == ret) update_h_ttrack (gb, tgt);
	return ret;
}

static void copy_scratchpad_to_multi_tgt (gpu_calc *gb, token_tracking *tgt, token_tracking *scratchpad, const unsigned int num);
static void copy_scratchpad_to_multi_tgt (gpu_calc *gb, token_tracking *tgt, token_tracking *scratchpad, const unsigned int num)
{
	// since we increased it in mv_one_chunk_left: called before this func
	scratchpad -> h_num_free -= tgt -> chunk_size;

	for (unsigned i = 0; i < num && tgt; i++, tgt = tgt -> next) {
		copy_scratchpad_to_tgt (gb, tgt, scratchpad, i);
		// Dbg ("------ tgt %u inside copy_scratchpad_to_multi_tgt --------", i); print_token_from_device (gb, tgt);
	}
}
/************* scratchpad_to_tgt functions ***************/

/*************** implementation ***************
 * cp src to scr
 * while (src) cp src -> scr
 * 		sort scr, cp scr -> tgt
 * 		shift scr left -repeat till src emptied
 * end_while
 * cp balance scr -> tgt
 *************** h_num_free ***************
 * start: scr -> mark as empty
 * copy_src_to_scratchpad:
 * 		reduce scr->free by src->(size - free)
 * sort scr - update scr-> free
 * copy_scratchpad_to_tgt:
 * 		if scr->free >= tgt-> size then don't copy anything to tgt; next(tgt)
 * 		if (scr->free) tgt->free = scr-> free; cp data scr->tgt
 * 		if (!scr->free) tgt->free = scr-> free; cp data scr->tgt
 * mv_one_chunk_left:
 * 		increase scr->free by tgt->size;
 ***/

static bool sort_merge_histo_cross (all_bufs *ab, token_tracking **curr , data_sort_otc *d_sotc)
{
	if (!(*curr) -> prev && !(*curr) -> next) return true;
	static unsigned int		max_scr_chunks = 8;	// 8 chunks max
	static token_tracking *scratchpad = create_token_tracking (&ab -> gc, max_scr_chunks * ab -> def_gpu_calc.block_size);

	data_sort_otc hd_sotc;
	gpuErrChk (cudaMemcpy (&hd_sotc, d_sotc, sizeof (data_sort_otc), cudaMemcpyDeviceToHost));	// data now in cpu
	unsigned int tot_chunks = 0;
	token_tracking *src = ab -> h_ttrack;
	for (token_tracking *tmp = src; tmp; tmp = tmp -> next) tot_chunks++;
	gpu_calc gb;
	memcpy (&gb, &ab -> def_gpu_calc, sizeof (gpu_calc));

	if (max_scr_chunks < tot_chunks) tot_chunks = max_scr_chunks;
	hd_sotc.n_chunks = tot_chunks;
	hd_sotc.n_warps = tot_chunks * ab -> def_gpu_calc.block_size / gb.warp_size;
	gpuErrChk (cudaMemcpy (d_sotc, &hd_sotc, sizeof (data_sort_otc), cudaMemcpyHostToDevice));	// data now in GPU

	for (src = ab -> h_ttrack; src && (src -> chunk_size == src -> h_num_free); src = src -> next);			// if empty, next
	if (!src || !ab -> h_ttrack -> next) {	// all empty or only one chunk
		set_sort_true (d_Srt);
		return true;
	}

	// reset scratchpad everytime this func is called
	gpuErrChk (cudaMemset (scratchpad -> d_data, '\0', sizeof (token *) * scratchpad -> chunk_size));
	scratchpad -> h_num_free = scratchpad -> chunk_size = tot_chunks * gb.block_size;
	gpuErrChk (cudaMemcpy (scratchpad -> d_num_free, &scratchpad -> h_num_free, sizeof (unsigned int), cudaMemcpyHostToDevice));

	token_tracking *tgt = src;
	bool isEmptyToken = false;
	src = copy_multi_src_to_scratchpad (scratchpad, src, tot_chunks - 1);
	// Dbg ("\n\n------ after copy_multi_src_to_scratchpad -------"); print_scratchpad_and_chunks (&gb, scratchpad, src);

	// unsigned int i = 0;
	for (; src; src = src -> next) {
		copy_src_to_scratchpad (scratchpad, tot_chunks - 1, src);	// last src

		// Dbg ("\n\n------ after copy_src_to_scratchpad -------"); print_scratchpad_and_chunks (&gb, scratchpad, src);

		scratchpad -> h_num_free = sort_one_token_block (&gb, scratchpad, d_sotc);	// sort
		// Dbg ("------ after sort_one_token_block, print src --------"); print_scratchpad_and_chunks (&gb, scratchpad, src);

		// cp to target
		if (!copy_scratchpad_to_tgt (&gb, tgt, scratchpad)) isEmptyToken = true;
		// Dbg ("------ after copy_scratchpad_to_tgt, print tgt --------"); print_scratchpad_and_chunks (&gb, scratchpad, tgt);
		tgt = tgt -> next;

		mv_one_chunk_left (scratchpad, tot_chunks, tgt -> chunk_size);
		// Dbg ("------ scratchpad after mv_one_chunk_left --------"); PRINT_HISTO (&gb, scratchpad);
		// Dbg ("i %u", ++i);
	}

	copy_scratchpad_to_multi_tgt (&ab -> def_gpu_calc, tgt, scratchpad, tot_chunks - 1);

	/***
	Dbg ("\n\n------ after copy_scratchpad_to_multi_tgt -------");
	token_tracking *tst = ab -> h_ttrack;	// RS_DEBUG
	while (tst && tst -> prev) tst = tst -> prev;
	for (; tst; tst = tst -> next) print_token_from_device (&gb, tst);
	for (; tst; tst = tst -> next) PRINT_HISTO (&gb, tst);
	***/

	return isEmptyToken;		// based on this we will compress out empty tokens
}
/************* end sort_merge_histo_cross functions ***************/
