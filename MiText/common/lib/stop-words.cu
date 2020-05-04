/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * stop-words.cu
 ******************************************************/

#include <cuda_milib.h>
#include <milib.h>

/*************  Imported Functions  **************/
extern void set_to_zero (unsigned int *d_val);
extern unsigned int cuda_stream_to_wd_to_token (ssize_t read, all_bufs *ab);
// don't merge, just sort
extern void milib_gpu_sort_merge_histo_wds (all_bufs *ab, bool is_stop_words);
extern void reset_all_tt (all_bufs *ab);
extern __device__ void zero_out_token (token *curr, unsigned int *num_free);
extern bool __device__ __host__ isGreater (const mhash_vals *curr, const mhash_vals *next);
extern bool __device__ __host__ isEqual (const mhash_vals *curr, const mhash_vals *next);
// extern __device__ void cuda_swap_mem (void *tmp_mem,  void *d_mem1, void *d_mem2, const unsigned int ln); for testing

/*************  Exported Functions  **************/
void create_stop_word (all_bufs *ab, FILE *fl);
void apply_stop_words (token_tracking *hs, all_bufs *ab);

// test functions
extern void print_token_tracking (gpu_calc *gb, token_tracking *hst);
void print_stop_words (gpu_calc *gb, mhash_vals *d_list, unsigned int *d_nos);
extern void print_token_from_device (gpu_calc *gb, token_tracking *tt);	// for testing RS_DEBUG

static __global__ void K_histo_hash_to_sw_hash (mhash_vals *tgt_sw, token **src_sw, const unsigned int max_sw);
static __global__ void K_histo_hash_to_sw_hash (mhash_vals *tgt_sw, token **src_sw, const unsigned int max_sw)
{
	const unsigned int	gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= max_sw) return;
	if (NULL == src_sw [gTh]) return;

	cuda_MemCpy (&(tgt_sw [gTh]), &(src_sw [gTh] -> mhv), sizeof (mhash_vals));
	// CudaDbgPrn ("gTh %u hash %u %.2s", (unsigned) gTh, (unsigned) tgt_sw [gTh].mhash, tgt_sw [gTh].str);
}

static void cuda_create_stop_word_lst (all_bufs *ab, unsigned int *d_wds);
static void cuda_create_stop_word_lst (all_bufs *ab, unsigned int *d_wds)
{
	// create/update d_stop_word_nos
	if (!ab -> d_sw_nos) {
		gpuErrChk (cudaMalloc (&(ab -> d_sw_nos), sizeof (unsigned int)));
		gpuErrChk (cudaMemcpy (ab -> d_sw_nos, d_wds, sizeof (unsigned int), cudaMemcpyDeviceToDevice));
	}

	// create d_stop_words mem
	unsigned int h_wds;
	gpuErrChk (cudaMemcpy (&h_wds, d_wds, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	if (!ab -> d_sw_list) {
		gpuErrChk (cudaMalloc (&(ab -> d_sw_list), sizeof (mhash_vals) * h_wds));
	}

	unsigned int start = 0;
	dim3 grid ((ab -> def_gpu_calc.block.x + h_wds - 1) / ab -> def_gpu_calc.block.x);
	for (token_tracking *tmp = ab -> h_ttrack; tmp; tmp = tmp -> next) {
		K_histo_hash_to_sw_hash <<<grid, ab -> def_gpu_calc.block>>> (
			ab -> d_sw_list + start, tmp -> d_data + tmp -> h_num_free,	// tgt, src mem
			tmp -> chunk_size - tmp -> h_num_free);		// how much to cpy
		CUDA_GLOBAL_CHECK;

		start += (tmp -> chunk_size - tmp -> h_num_free);
	}

	reset_all_tt (ab);
	Dbg ("After reset_all_tt"); print_stop_words (&ab -> def_gpu_calc, ab -> d_sw_list, ab -> d_sw_nos);
}

/****
 * Assumption: We can read all the stop words in one shot.  We don't need
 * to implement the huge complex cuda_read_buffer_into_gpu and it's related
 * functions.
 ****/
static ssize_t cuda_stop_word_read_buffer_into_gpu (all_bufs *ab, FILE *f);
static ssize_t cuda_stop_word_read_buffer_into_gpu (all_bufs *ab, FILE *f)
{
	ssize_t	chars_read;

	chars_read = (ssize_t) fread (ab -> st_info.h_read_buf,  (size_t) sizeof (unsigned char), (size_t) ab -> st_info.bufsiz - 1, f);
	ab -> st_info.h_read_buf [chars_read] = '\0';
	if (0 > chars_read) {
		Dbg ("Stop word Read err %d", (int) chars_read);
		exit (chars_read);
	}

	for (;!cuda_AsciiIsAlnumApostrophe (ab -> st_info.h_read_buf [chars_read]); chars_read--) {
		ab -> st_info.h_read_buf [chars_read] = '\0';
	}			// go back if the last chars are not alpha numeric
	chars_read++;

	gpuErrChk (cudaMemcpy (ab -> st_info.d_curr_buf, ab -> st_info.h_read_buf, (chars_read + 1) * sizeof (unsigned char), cudaMemcpyHostToDevice));
	Dbg ("cp'd to device - chars %d line %.100s...", (int) chars_read, ab -> st_info.h_read_buf);

	return chars_read;
}

// read from stdin
void create_stop_word (all_bufs *ab, FILE *fl)
{
	ssize_t			c_read;

	ab -> IsS_wd = true;
	c_read = cuda_stop_word_read_buffer_into_gpu (ab, fl);
	Dbg ("chars %u: stopwords read:", (unsigned) c_read);

	ab -> h_Swds = cuda_stream_to_wd_to_token (c_read, ab);
	if (ab -> h_Swds) {
		milib_gpu_sort_merge_histo_wds (ab, true);	// is_stop_words?
	}

	cuda_create_stop_word_lst (ab, ab -> d_wds);
	set_to_zero (ab -> d_wds);
	Dbg ("======== done ========");
}

// RS_DEBUG - performance IS pathetic - to relook in future.
// TBD - Alt 1: store stop words in texture memory
// TBD - Alt 2: mv stop words to shmem & then apply from there
static __global__ void K_Histo_apply_stop_words (token **tok, const mhash_vals *stp_wds, const unsigned int *swd_nos, unsigned int *num_free, const unsigned int c_size);
static __global__ void K_Histo_apply_stop_words (token **tok, const mhash_vals *stp_wds, const unsigned int *swd_nos, unsigned int *num_free, const unsigned int c_size)
{
	extern __shared__ tok_mhash chunk [];		// shmem of d_data
	const unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh < c_size) {
		if (tok [gTh]) {cuda_MemCpy (&(chunk [gTh].mhv), &(tok [gTh] -> mhv), sizeof (mhash_vals));}
		else {cuda_MemSet (&chunk [gTh], (uint8_t) 0x00, (uint32_t) sizeof (tok_mhash));}
	}
	__threadfence ();
	__syncthreads ();

	if (*swd_nos <= gTh) return;

#pragma unroll 128
	for (int i = 0; i < c_size; i++) {
		// TBD: RS_DEBUG relook = each thread should be one combo of chunk & swd. texture memory?

		// CudaDbgPrn ("stp:%u %u %.2s|token_loc:%u %u %.2s", (unsigned) gTh, (unsigned) stp_wds [gTh].mhash, stp_wds [gTh].str, (unsigned) i, (unsigned) tok [i] -> mhv.mhash, tok [i] -> mhv.str);

		// CudaDbgPrn ("curr-tok/next-stp_wds: ");
		if (true == isEqual (&(chunk [i].mhv), &(stp_wds [gTh]))) {
			// CudaDbgPrn ("**Zeroing: token_loc:sw_loc %u:%u: chunk:mhash %u %.2s", (unsigned) i, (unsigned) gTh, (unsigned) chunk [i].mhv.mhash, chunk [i].mhv.str);

			tok [i] = (token *) NULL;
			cuda_MemSet (&chunk [i], (uint8_t) 0x00, (uint32_t) sizeof (tok_mhash));
			atomicAdd ((unsigned int *) num_free, (unsigned int) 1);
		}
	}
}

void apply_stop_words (token_tracking *hs, all_bufs *ab)
{
	static unsigned int h_num_sword = 0;

	if (true == hs -> stopWord_applied) return;
	if (!h_num_sword) {
		gpuErrChk (cudaMemcpy (&h_num_sword, ab -> d_sw_nos, sizeof (unsigned int), cudaMemcpyDeviceToHost));
		if (!h_num_sword) {
			Dbg ("Error: No Stop Word, exiting");
			exit (1);
		}
	}

	// Dbg ("To apply stop words");
	// dim3 block (ab -> def_gpu_calc.block_size, ab -> def_gpu_calc.block_size);
	dim3 block (hs -> chunk_size);
	dim3 grid ((ab -> h_Swds + block.x - 1) / block.x);

	// apply and delete sw
	/***
	Dbg ("*** Applying stop words ***");
	Dbg ("stop_words"); print_stop_words (&ab -> def_gpu_calc, ab -> d_sw_list, ab -> d_sw_nos);
	Dbg ("To list:"); PRINT_HISTO (&ab -> def_gpu_calc, hs);
	***/

	K_Histo_apply_stop_words <<<grid, block, hs -> chunk_size * sizeof (tok_mhash)>>> (hs -> d_data, ab -> d_sw_list, ab -> d_sw_nos, hs -> d_num_free, hs -> chunk_size);
	CUDA_GLOBAL_CHECK;
	// Dbg ("Applied stop words");
	hs -> stopWord_applied = true;			// stop_word_applied
}
