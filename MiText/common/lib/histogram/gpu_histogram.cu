/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * gpu_histogram.cu
 ******************************************************/

#include <milib.h>
#include <cuda_milib.h>

// functions exported
unsigned int process_next_set_of_tokens (all_bufs *ab, ssize_t data_read);
void cuda_read_buffer_into_gpu (all_bufs *ab, unsigned chars_read);
unsigned int cuda_stream_to_wd_to_token (ssize_t read, all_bufs *ab);
token_tracking *delete_token_tracking (token_tracking *curr);
void cuda_free_everything (all_bufs *ab);
token_tracking *create_token_tracking (gpu_config *gc, const unsigned int c_size);

void histo_initialize_global_structs (all_bufs *ab, unsigned int wds);
void update_h_ttrack (gpu_calc *gb, token_tracking *h_ttrack);
void __global__ K_get_d_data_out (token *out, token **d_data, const size_t c_size);
void reset_all_tt (all_bufs *ab);
void memory_compaction (all_bufs *ab);

// functions imported
extern __device__ uint32_t MurmurHash3_32 (const unsigned char *data, const unsigned int nbytes);
extern void milib_gpu_sort_merge_histo_wds (all_bufs *ab, bool merge);
extern void create_stop_word (all_bufs *ab, FILE *fl);
extern void set_to_zero (unsigned int *d_val);
extern unsigned int get_d_val (unsigned int *d_val);

// debug
extern void print_token_from_device (gpu_calc *gb, token_tracking *tt);

// local functions, struct & variables
// used in cuda_read_buffer_into_gpu
typedef struct past_c_buf {
	unsigned char	pbuf [STAT_MAX_WORD_SIZE];
	unsigned int	len;
} past_c_buf;

// __device__ unsigned int	d_wds [1]	= {0};	// number of  words

static __global__ void K_histo_stream_to_words (unsigned char *d_buf, unsigned int *d_loc, ssize_t read, unsigned int *wd_idx);

// We assume that we will not go out of memory.  Check for mem availability here and plan recovery from OOM situation.
unsigned int process_next_set_of_tokens (all_bufs *ab, ssize_t data_read)
{
	unsigned int words = cuda_stream_to_wd_to_token (data_read, ab);	// tokenize

	if (!words) return words;
	// Dbg ("data_read %u", (unsigned) data_read);
	milib_gpu_sort_merge_histo_wds (ab, false);	// sort & merge on crc
	return words;
}

// created because we didn't have sufficient memory
typedef struct MY_ALIGN (16) data_for_K_words_to_token {
	unsigned char	*d_wds;
	unsigned int	*d_loc;
	unsigned int	*wd_idx;				// total words
	size_t			loc;					// start loc of words
	size_t			sz;						// chunk_size
} data_for_K_words_to_token;

static __global__ void K_words_to_token (token **d_data, const data_for_K_words_to_token *kw, unsigned int *d_free);

unsigned int cuda_stream_to_wd_to_token (ssize_t read, all_bufs *ab)
{
	dim3 block = ab -> def_gpu_calc.block_size;
	dim3 grid	((read + block.x - 1) / block.x);

	set_to_zero (ab -> d_wds);
	K_histo_stream_to_words <<<grid, block>>>
		(ab -> st_info.d_curr_buf,			// raw data
		ab -> st_info.d_loc,				// loc of start of wd - filled here
		read,								// tot chars read
		ab -> d_wds);						// tot words found
	CUDA_GLOBAL_CHECK;

	const unsigned int new_wds = get_d_val (ab -> d_wds);
	if (!new_wds) return new_wds;
	ab -> h_math.h_tot_wds_pre_sw += new_wds;

	const unsigned int new_hist = cuda_int_div_rounded (new_wds, ab -> def_gpu_calc.block_size);
	Dbg ("data_read %u words %u hist %u", (unsigned) read, (unsigned) new_wds, (unsigned) new_hist);

	token_tracking *tmp_hist = ab -> h_ttrack, *start;
	if (tmp_hist) {
		while (tmp_hist -> next) tmp_hist = tmp_hist -> next;
		tmp_hist -> next = create_token_tracking (&ab -> gc, ab -> def_gpu_calc.block_size);
		tmp_hist -> next -> prev = tmp_hist;
		start = tmp_hist = tmp_hist -> next;
	}
	else {
		start = tmp_hist = ab -> h_ttrack = create_token_tracking (&ab -> gc, ab -> def_gpu_calc.block_size);
	}

	// one already created before so new_hist - 1
	for (unsigned int i = 0; i < (new_hist - 1); i++) {	// create blank tt AFTER
		tmp_hist -> next = create_token_tracking (&ab -> gc, ab -> def_gpu_calc.block_size);
		tmp_hist -> next -> prev = tmp_hist;
		tmp_hist = tmp_hist -> next;
		// Dbg ("Token tracking created");
	}
	tmp_hist = start;					// put tokens from here

	data_for_K_words_to_token hd_Kwtt;

	static data_for_K_words_to_token *d_Kwtt = NULL;
	if (!d_Kwtt) {
		gpuErrChk (cudaMalloc (&d_Kwtt, sizeof (data_for_K_words_to_token)));
	}
	hd_Kwtt.d_wds = ab -> st_info.d_curr_buf;
	hd_Kwtt.d_loc = ab -> st_info.d_loc;
	hd_Kwtt.wd_idx = ab -> d_wds;

	unsigned int sh_memry = ab -> gc.shmem / sizeof (token);
	if (tmp_hist -> chunk_size < sh_memry) sh_memry = tmp_hist -> chunk_size;

	dim3 wd_grid ((new_wds + block.x - 1) / block.x);
	for (unsigned int i = 0; tmp_hist && i < new_hist; tmp_hist = tmp_hist -> next, i++) {
		hd_Kwtt.loc = i * tmp_hist -> chunk_size;
		hd_Kwtt.sz = tmp_hist -> chunk_size;
		gpuErrChk (cudaMemcpy (d_Kwtt, &hd_Kwtt, sizeof (data_for_K_words_to_token), cudaMemcpyHostToDevice));

		// Dbg ("before K_words_to_token, i %u, loc %u", (unsigned) i, (unsigned) (i * tmp_hist -> chunk_size));

		K_words_to_token <<<wd_grid, block, sh_memry * sizeof (token)>>>
			(tmp_hist -> d_data,		// token to fill out
			d_Kwtt,						// static stuff
			tmp_hist -> d_num_free);	// free space in this d_data
		CUDA_GLOBAL_CHECK;

		// we don't HAVE to over here but for Integrity's sake...
		gpuErrChk (cudaMemcpy (&tmp_hist -> h_num_free, tmp_hist -> d_num_free, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	}

	Dbg ("read:%u, words:%u str %.100s\n", (unsigned) read, new_wds, ab -> st_info.h_read_buf);

	return new_wds;
}

void cuda_read_buffer_into_gpu (all_bufs *ab, unsigned chars_read)
{
	gpuErrChk (cudaMemcpy (ab -> st_info.d_curr_buf, ab -> st_info.h_read_buf, chars_read * sizeof (unsigned char), cudaMemcpyHostToDevice));
	// Dbg ("cp'd to device - chars %d line %1000s...", (int) chars_read, ab -> st_info.h_read_buf);

	ssize_t	h_loc_maxsz;
	gpuErrChk (cudaMemcpy (&h_loc_maxsz, ab -> st_info.d_loc_maxsz, sizeof (size_t), cudaMemcpyDeviceToHost));
	if (h_loc_maxsz < chars_read) {	// possible words read < max chars read
		CUDA_FREE (ab -> st_info.d_loc);

		h_loc_maxsz = chars_read * 1.2;	// 20% more
		gpuErrChk (cudaMalloc (&(ab -> st_info.d_loc), sizeof (size_t) * h_loc_maxsz));
		gpuErrChk (cudaMemcpy (ab -> st_info.d_loc_maxsz, &h_loc_maxsz, sizeof (size_t), cudaMemcpyHostToDevice));
		gpuErrChk (cudaMemset (ab -> st_info.d_loc, '\0', sizeof (size_t) * h_loc_maxsz));
	}
}

static void add_gpuMem_to_token_tracking (gpu_config *gc, token_tracking *tt, const unsigned int tok_size);
static void fill_out_token_tracking (token_tracking *tt, unsigned int tok_size);

/***
 * Default token size is chunk_size.  In this case block_size == chunk_size
 * in case of scratchpad token size is block_size which consists of multiple
 * chunks
 ***/
token_tracking *create_token_tracking (gpu_config *gc, const unsigned int c_size)
{
	token_tracking	*h_new = (token_tracking *) malloc (sizeof (token_tracking));
	check_mem (h_new);
	memset (h_new, '\0', sizeof (token_tracking));

	add_gpuMem_to_token_tracking (gc, h_new, c_size);
	h_new -> h_data	= (token *) malloc (sizeof (token) * c_size);
	check_mem (h_new -> h_data);

	fill_out_token_tracking (h_new, c_size);
	return h_new;

error:
	Dbg ("Out of Memory, exiting");
	exit (1);
}

static void phy_delete_tt_chunk (token_tracking *curr);

// we delete the tt chunk iff all tokens are free
// we return previous chunk if it exists otherwise the next chunk
token_tracking *delete_token_tracking (token_tracking *curr)
{
	token_tracking *to_ret;
	token_tracking *prev, *next;
	if (!curr) return NULL;

	prev = curr -> prev;	// grab prev and next
	next = curr -> next;

	// curr = prev, del curr,
	if (!next && prev) {			// next doesn't exist, this is last
		// prev exists
		to_ret = prev;
		to_ret -> next = NULL;
	}
	else if (next && !prev) {		// del curr, reset next, return next
		to_ret = next;
		to_ret -> prev = NULL;
	}
	else if (!next && !prev) {
		to_ret = NULL;
	}
	// curr = prev, del next prev = next-> next next = prev
	else {					// next & prev exist, this is middle
		to_ret = next;
		to_ret -> prev = prev;
		to_ret -> next = next -> next;
	}
	phy_delete_tt_chunk (curr);

	return to_ret;
}

static void phy_delete_tt_chunk (token_tracking *curr)
{
	token_tracking	*todel = curr;

	CUDA_FREE (todel -> d_num_free);
	CUDA_FREE (todel -> d_data);
	CUDA_FREE (todel -> d_tok_start);
	FREE (todel -> h_data);
	FREE (todel);
}

// delete all tt except one.
// ensure that this tt is flushed out
void reset_all_tt (all_bufs *ab)
{
	token_tracking *tmp_tt = ab -> h_ttrack;
	while (tmp_tt) tmp_tt = delete_token_tracking (tmp_tt);
	ab -> h_ttrack = NULL;
}

void cuda_free_everything (all_bufs *ab)
{
	FREE (ab -> st_info.h_read_buf);
	FREE (ab -> h_math.h_num_list);
	reset_all_tt (ab);	// free every h_ttrack

	// no CUDA_FREE done as all mem destroyed by cudaDeviceReset
	gpuErrChk (cudaDeviceReset ());		// reset device
}

/****************  Histogram Functions  ********************/
static __global__ void K_histo_stream_to_words (unsigned char *d_buf, unsigned int *d_loc, ssize_t read, unsigned int *wd_idx)
{
	const int	gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (int) read) return;

	const bool	is_ch = cuda_AsciiIsAlnumApostrophe (d_buf [gTh]);	// curr char
	// CudaDbgPrn ("th:%d wd_idx%u read:%u:is_ch %d char %c %d", gTh, (unsigned) *wd_idx, (unsigned) read, (int) is_ch, d_buf [gTh], (int) d_buf [gTh]);
	if (!is_ch) d_buf [gTh] = (unsigned char) '\0';
	if (cuda_AsciiIsAlpha (d_buf [gTh])) d_buf [gTh] |= 0x20;	// lower case

	bool	is_prev_ch = false;	// 1st char is false; don't do AsciiIsAlnum at gTh == 0
	if (((int) 0) != gTh) {		// for rest of the chars
		is_prev_ch = cuda_AsciiIsAlnumApostrophe (d_buf [gTh - 1]);	// prev char
	}

	if (is_ch && !is_prev_ch) {	// beginning of word
		unsigned int old_wd = atomicAdd ((unsigned int *) wd_idx, (unsigned int) 1);
		d_loc [old_wd] = gTh;		// loc of start of next word
		// CudaDbgPrn ("th (& loc): %u old_wd (wd no) %u: wd %.20s", (unsigned) gTh, (unsigned) old_wd, &d_buf [gTh]);
	}
}

static __global__ void K_words_to_token (token **d_data, const data_for_K_words_to_token *kw, unsigned int *d_free)
{
	const int	gTh = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ token st [];
	const int	d_loc_idx = kw -> loc + gTh;

	if (gTh >= (int) kw -> sz) return;
	cuda_MemSet (&st [gTh], (uint8_t) '\0', (uint32_t) sizeof (token));

	if (d_loc_idx < *kw -> wd_idx) {	// d_loc_idx is the word num
		// CudaDbgPrn ("gTh %u:mem %p: chunk no %u d_loc_idx:%u free %u", (unsigned) gTh, d_data [gTh], (unsigned) kw -> loc, (unsigned) d_loc_idx, (unsigned) *d_free);

		unsigned len = (unsigned) cuda_string_len ((unsigned char *) &kw -> d_wds [kw -> d_loc [d_loc_idx]]);
		if (len > STAT_MAX_WORD_SIZE) len = (unsigned) STAT_MAX_WORD_SIZE - 2;
		// CudaDbgPrn ("gTh: %u before dbuf %.20s len %u", (unsigned) gTh, &kw -> d_wds [kw -> d_loc [d_loc_idx]], (unsigned) len);

		cuda_MemCpy (st [gTh].wd.str, (void *) &kw -> d_wds [kw -> d_loc [d_loc_idx]], (uint32_t) (len * sizeof (unsigned char)));
		st [gTh].wd.str [len]		= (unsigned char) '\0';
		st [gTh].wd.str [len + 1]	= (unsigned char) '\0';
		st [gTh].wd.len				= (unsigned int) len;
		st [gTh].wd.num				= (uint32_t) 1;
		// CudaDbgPrn ("wd th:%u: str %.20s, len %u, num %u", (unsigned) gTh, st [gTh].wd.str, (unsigned) st [gTh].wd.len, (unsigned) st [gTh].wd.num);

		st [gTh].mhv.mhash	= (uint32_t) MurmurHash3_32 (st [gTh].wd.str, (unsigned int) len);
		// CudaDbgPrn ("gTh: %u: hash %u, str %.20s len %u", (unsigned) gTh, (unsigned) st [gTh].mhv.mhash, st [gTh].wd.str, (unsigned) st [gTh].wd.len);

		st [gTh].mhv.str [0]		= (unsigned char) st [gTh].wd.str [0];
		if (1 < len) st [gTh].mhv.str [1]	= (unsigned char) st [gTh].wd.str [1];
		else st [gTh].mhv.str [1]	= (unsigned char) '\0';
	}
	__syncthreads ();
	// CudaDbgPrn ("st: gTh: %u: hash %u, chars %.2s", (unsigned) gTh, (unsigned) st [gTh].mhv.mhash, st [gTh].mhv.str);

	if (st [gTh].mhv.mhash) {
		cuda_MemCpy (d_data [gTh], &st [gTh], sizeof (token));
		atomicSub (d_free, (unsigned int) 1);
		// CudaDbgPrn ("d_data: gTh: %u: hash %u, chars %.2s free %u", (unsigned) gTh, (unsigned) d_data [gTh] -> mhv.mhash, d_data [gTh] -> mhv.str, (unsigned int) *d_free);
	}
	else {
		d_data [gTh] = (token *) NULL;
	}
}

// initialize the global vars for next set words coming in
void histo_initialize_global_structs (all_bufs *ab, unsigned int wds)
{
	set_to_zero (ab -> d_wds);
}

static void __global__ K_add_tok_mem_to_d_data (token **d_data, token *d_contents, const unsigned int tok);
static void add_gpuMem_to_token_tracking (gpu_config *gc, token_tracking *tt, const unsigned int tok_size)
{
	token *d_dt;		// pointers in d_data
	gpuErrChk (cudaMalloc (&d_dt, sizeof (token) * tok_size));
	gpuErrChk (cudaMemset (d_dt, '\0', sizeof (token) * tok_size));
	tt -> d_tok_start = d_dt;

	token **d_ret;		// new d_data
	gpuErrChk (cudaMalloc (&d_ret, sizeof (token **) * tok_size));
	gpuErrChk (cudaMemset (d_ret, '\0', sizeof (token **) * tok_size));

	dim3 block (tok_size);
	dim3 grid ((tok_size * block.x - 1) / block.x);

	K_add_tok_mem_to_d_data <<<block, grid>>> (d_ret, d_dt, tok_size);
	CUDA_GLOBAL_CHECK;

	tt -> d_data = d_ret;
}

static void __global__ K_add_tok_mem_to_d_data (token **d_data, token *d_contents, const unsigned int tok)
{
	const unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (unsigned int) tok) return;

	d_data [gTh] = (token *) (d_contents + gTh);
}

static void fill_out_token_tracking (token_tracking *tt, unsigned int tok_size)
{
	tt -> h_num_free = tok_size;
	tt -> chunk_size = tok_size;
	tt -> stopWord_applied = false;
	tt -> prev = tt -> next = NULL;

	gpuErrChk (cudaMalloc (&(tt -> d_num_free), sizeof (unsigned int)));
	gpuErrChk (cudaMemcpy (tt -> d_num_free, &(tt -> h_num_free), sizeof (unsigned int), cudaMemcpyHostToDevice));
}

// This is done only when there are empty tokens or at end of sorting
// memory compaction done ONLY after a complete sort so the zeros are
// all on the left.
static void cp_src_to_tgt (gpu_calc *gb, token_tracking *dest, token_tracking *src);
void memory_compaction (all_bufs *ab)
{
	token_tracking *tt_lst = ab -> h_ttrack;
	while (tt_lst -> prev) tt_lst = tt_lst -> prev;	// go to head

	gpu_calc gb;
	gb.block = ab -> def_gpu_calc.block;
	token_tracking	*n_hd = NULL, *tgt_tt = NULL;
	for (token_tracking *src_tt = tt_lst; src_tt; src_tt = src_tt -> next) {
		token_tracking *tmp;

		if (src_tt -> chunk_size == src_tt -> h_num_free) {	// empty
			if (!src_tt -> prev && !src_tt -> next) {		// only one src
				tmp = create_token_tracking (&ab -> gc, ab -> def_gpu_calc.block_size);
				n_hd = tgt_tt = tmp;
			}
		}
		else {												// not empty
			tmp = create_token_tracking (&ab -> gc, ab -> def_gpu_calc.block_size);
			if (!tgt_tt) {
				n_hd = tgt_tt = tmp;
			}
			else {
				tgt_tt -> next = tmp;
				tmp -> prev = tgt_tt;
				tgt_tt = tmp;
			}

			cp_src_to_tgt (&gb, tgt_tt, src_tt);
			update_h_ttrack (&gb, tgt_tt);
		}
	}

	for (token_tracking *old = ab -> h_ttrack; old;) {	// delete orig. h_ttrack
		old = delete_token_tracking (old);
	}

	ab -> h_ttrack = n_hd;			// replace orig h_ttrack with n_hd
	Dbg ("compressed all the token tracking");
}

static __global__ void K_cp_src_to_tgt (token **dest, token **src, const unsigned int len);
static __global__ void K_cp_src_to_tgt (token **dest, token **src, const unsigned int len)
{
	const unsigned int	gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (unsigned int) len) return;

	if (src [gTh] && src [gTh] -> mhv.mhash) cuda_MemCpy (dest [gTh], src [gTh], (uint32_t) sizeof (token));
	else (cuda_MemSet (dest [gTh], (uint8_t) '\0', (uint32_t) sizeof (token)));

	/***
	if (dest [gTh]) {
		CudaDbgPrn (": th%u dest mhash %u str %.20s", (unsigned) gTh, (unsigned) dest [gTh] -> mhv.mhash, dest [gTh] -> wd.str);
	}
	else {
		CudaDbgPrn (": dest [%u] (null)", (unsigned) gTh);
	}
	***/
}

static void cp_src_to_tgt (gpu_calc *gb, token_tracking *dest, token_tracking *src)
{
	dim3 grid ((src -> chunk_size + gb -> block.x - 1) / gb -> block.x);
	K_cp_src_to_tgt <<<grid, gb -> block>>> (dest -> d_data, src -> d_data, dest -> chunk_size);
	CUDA_GLOBAL_CHECK;

	dest -> stopWord_applied = src -> stopWord_applied;
	dest -> h_num_free = src -> h_num_free;
	gpuErrChk (cudaMemcpy (dest -> d_num_free, src -> d_num_free, sizeof (unsigned int), cudaMemcpyDeviceToDevice));
}

// should not call memory_compaction from inside this as it will be a forever loop.
void update_h_ttrack (gpu_calc *gb, token_tracking *h_ttrack)
{
	static token		*d_out = NULL;
	static unsigned int	sz = 0x0;
	if (0x0 == sz) sz = h_ttrack -> chunk_size;
	dim3 grid ((h_ttrack -> chunk_size + gb -> block.x - 1) / gb -> block.x);

	if (!d_out) {
		gpuErrChk (cudaMalloc (&d_out, sizeof (token) * h_ttrack -> chunk_size));
	}
	else if (sz < h_ttrack -> chunk_size) {
		CUDA_FREE (d_out);
		gpuErrChk (cudaMalloc (&d_out, sizeof (token) * h_ttrack -> chunk_size));
		sz = h_ttrack -> chunk_size;
	}

	gpuErrChk (cudaMemset (d_out, '\0', sizeof (token) * h_ttrack -> chunk_size));
	gpuErrChk (cudaMemcpy (&h_ttrack -> h_num_free, h_ttrack -> d_num_free, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	if (h_ttrack -> chunk_size > h_ttrack -> h_num_free) {
		K_get_d_data_out <<<grid, gb -> block>>> (d_out, h_ttrack -> d_data, h_ttrack -> chunk_size);
		CUDA_GLOBAL_CHECK;
	}

	gpuErrChk (cudaMemcpy (h_ttrack -> h_data, d_out, sizeof (token) * h_ttrack -> chunk_size, cudaMemcpyDeviceToHost));
	h_ttrack -> first_hash = h_ttrack -> h_data -> mhv.mhash;
}

void __global__ K_get_d_data_out (token *out, token **d_data, const size_t c_size)
{
	const unsigned int	gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (unsigned int) c_size) return;

	if (d_data [gTh]) cuda_MemCpy (&(out [gTh]), d_data [gTh], (uint32_t) sizeof (token));
	else cuda_MemSet (&(out [gTh]), (uint8_t) '\0', (uint32_t) sizeof (token));
}
