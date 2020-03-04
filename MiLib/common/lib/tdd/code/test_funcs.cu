/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * histo_balance.cu
 ******************************************************/

#include <cuda_milib.h>
#include <milib.h>

// this file contains all the functions needed for testing the histogram.

/******* Functions imported ************/
extern token_tracking *delete_token_tracking (const size_t sz, token_tracking *curr);
extern token_tracking *create_token_tracking (const unsigned int c_size);
extern void __global__ K_get_d_data_out (token *out, token **d_data, const size_t c_size);
extern void update_h_ttrack (gpu_calc *gb, token_tracking *h_ttrack);

extern __device__ volatile unsigned int	d_Srt [];	// high level sort

/******* Functions exported ************/
void print_token_from_device (gpu_calc *gb, token_tracking *tt);
void print_stop_words (gpu_calc *gb, mhash_vals *d_list, unsigned int *d_nos);
void print_token_tracking (gpu_calc *gb, token_tracking *hst);
void print_scratchpad_and_chunks (gpu_calc *gb, token_tracking *scr, token_tracking *src);

/******* Test Functions exported ************/
// RS_DEBUG - this test code is now obsolete. If we are going to do
// this testing again, uncommment the code and upgrade it
// wd - list of words is not enabled for this one.
// token_tracking *setup_tst_hist_sort_merge_histo_cross (token_tracking *hist);
/******* Local Functions ************/
// static void fill_out_hist_two (token_tracking *hs, const int goal);

static void __global__ K_Print_tok (const unsigned int c_size, token **tok, unsigned int *num_free);

static void __global__ K_Print_tok (const unsigned int c_size, token **tok, unsigned int *num_free)
{
	unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (unsigned int) c_size) return;

	if (tok [gTh]) {
		CudaDbgPrn ("gTh%u num:%u str %s hash%u free%u\n d_Srt %u", gTh, (unsigned) tok [gTh] -> wd.num, tok [gTh] -> wd.str, (unsigned) tok [gTh] -> mhv.mhash, *num_free, (unsigned) *d_Srt);
	}
	else {
		CudaDbgPrn ("gTh%u tok:NULL free%u\n d_Srt %u", gTh, *num_free, (unsigned) *d_Srt);
	}
}

static void __global__ K_Print_mhv (const unsigned int *c_size, mhash_vals *mhv);
static void __global__ K_Print_mhv (const unsigned int *c_size, mhash_vals *mhv)
{
	unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= (unsigned int) *c_size) return;

	CudaDbgPrn ("gTh%u hash:%u %.2s", gTh, (unsigned) mhv [gTh].mhash, mhv [gTh].str);
}

void print_token_from_device (gpu_calc *gb, token_tracking *tt)
{
	dim3 grid ((gb -> block_size + tt -> chunk_size - 1) / gb -> block_size);
	K_Print_tok <<<grid, gb -> block>>> (tt -> chunk_size, tt -> d_data, tt -> d_num_free);
	CUDA_GLOBAL_CHECK;
}

void print_stop_words (gpu_calc *gb, mhash_vals *d_list, unsigned int *d_nos)
{
	unsigned int sword_nos = 0;
	gpuErrChk (cudaMemcpy (&sword_nos, d_nos, sizeof (unsigned int), cudaMemcpyDeviceToHost));

	dim3 grid ((gb -> block_size * 2 - 1) / gb -> block_size);
	static mhash_vals *h_swds = NULL;
	if (!h_swds) {
		h_swds = (mhash_vals *) malloc (sizeof (mhash_vals) * sword_nos);
		check_mem (h_swds);
	}
	memset (h_swds, '\0', sizeof (mhash_vals) * sword_nos);

	Dbg ("====Printing Stop Wds: %u:", sword_nos);
	K_Print_mhv <<<grid, gb -> block>>> (d_nos, d_list);
	CUDA_GLOBAL_CHECK;

	gpuErrChk (cudaMemcpy (h_swds, d_list, sizeof (mhash_vals) * sword_nos, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < sword_nos; i++) {
		printf ("[%u]:%u %.2s\t", i, (unsigned) h_swds[i].mhash, h_swds[i].str);
		if (!(i % 4) && (i != 0)) printf ("\n");
	}
	printf ("\nPrinted Stop Words====\n");
	return;

error:
	Dbg ("Out of mem");
	exit (1);
}

void print_token_tracking (gpu_calc *gb, token_tracking *hst)
{
	if (!hst) {
		printf ("token tracking (null) - returning\n");
		return;
	}

	update_h_ttrack (gb, hst);

	token *tk = hst -> h_data;
	if (!tk) {
		printf ("token contents (null) - returning\n");
		return;
	}

	bool to_print_CR = false;
	printf ("\n");
	for (unsigned int i = 0; i < hst -> chunk_size; i++) {
		if (tk [i].mhv.mhash) {
			if (false == to_print_CR) printf ("\n");
			printf ("loc %u: str: %s: len: %u hash %u num %u\n", i, tk [i].wd.str, (unsigned) tk [i].wd.len, (unsigned) tk [i].mhv.mhash, (unsigned) tk [i].wd.num);
			if (false == to_print_CR) to_print_CR = true;
		}
		else {
			unsigned int j = 0;
			if (true == to_print_CR) {printf ("\n");} to_print_CR = false;
			printf ("%u (null)\t", i);
			if (!(++j % 15)) printf ("\n");
		}
	}

	printf ("\nnext %p prev %p stopWord_applied %u num_free: %u chunk_size %u first_hash %u\n", hst -> next, hst -> prev, (unsigned) hst -> stopWord_applied, hst -> h_num_free, hst -> chunk_size, hst -> first_hash);
}

void print_scratchpad_and_chunks (gpu_calc *gb, token_tracking *scr, token_tracking *tok)
{
	// we put part of update_h_ttrack code to get scr info out of the device
	static token *d_out = NULL;
	static unsigned int c_sz = 0;
	if (!c_sz) c_sz = scr -> chunk_size;
	if (!d_out) {
		gpuErrChk (cudaMalloc (&d_out, sizeof (token) * c_sz));
	}
	else if (d_out && c_sz < scr -> chunk_size) {
		CUDA_FREE (d_out);
		c_sz = scr -> chunk_size;
		gpuErrChk (cudaMalloc (&d_out, sizeof (token) * c_sz));
	}

	gpuErrChk (cudaMemset (d_out, '\0', sizeof (token) * scr -> chunk_size));
	gpuErrChk (cudaMemcpy (&scr -> h_num_free, scr -> d_num_free, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	if (scr -> chunk_size > scr -> h_num_free) {
		dim3 grid ((scr -> chunk_size * gb -> block.x - 1) / gb -> block.x);
		K_get_d_data_out <<<grid, gb -> block>>> (d_out, scr -> d_data, scr -> chunk_size);
		CUDA_GLOBAL_CHECK;

		gpuErrChk (cudaMemcpy (scr -> h_data, d_out, sizeof (token) * scr -> chunk_size, cudaMemcpyDeviceToHost));

		printf ("---------- Scratchpad ----------------\n");
		for (unsigned int i = 0; i < scr -> chunk_size; i++) {
			if (scr -> h_data [i].mhv.mhash) {
				printf ("scr->h_data[%u] %s len %u hash %u num %u\n", i, scr -> h_data[i].wd.str, (unsigned) scr -> h_data[i].wd.len, (unsigned) scr -> h_data[i].mhv.mhash, (unsigned) scr -> h_data[i].wd.num);
			}
			else printf ("scr->h_data[%u].mhv.mhash is null\n", i);
		}
		printf ("scr num_free %u\n", scr -> h_num_free);
	}
	else printf ("scr empty\n");

	if (tok) {
		printf ("-------- token --------\n");
		print_token_tracking (gb, tok);
	}
	else printf ("-------- token is (null) --------\n");
}

// These defines are related to the test case numbers
#define ONE_ONE_ONE			1
#define ONE_ONE_TWO			2
#define ONE_ONE_THREE		3
#define ONE_ONE_FOUR		4

// for 3 histo
#define ONE_TWO_ONE			5
#define ONE_TWO_TWO			6
#define ONE_TWO_THREE		7
#define ONE_TWO_FOUR		8
#define ONE_TWO_FIVE		9

/***
RS_DEBUG commented code is defective - create_token_tracking has been modified...
token_tracking *setup_tst_hist_sort_merge_histo_cross (token_tracking *hist, st_words *h_stp_wdsfl)
{
	// del all histo first
	token_tracking *tmp = hist;
	while (tmp -> next_chunk) tmp = delete_token_tracking (c_size, tmp);
	tmp -> d_st_stop = NULL;

	// two
	tmp -> next = create_token_tracking (c_size);
	tmp -> next -> prev = tmp;
	tmp = tmp -> next;

	// and one more
	tmp -> next = create_token_tracking (c_size);
	tmp -> next -> prev = tmp;
	tmp = tmp -> next;

	// 1.1.1 tmp val, nxt val
	// fill_out_hist_two (hist, ONE_ONE_ONE);
	// DBG_EXIT;

	// 1.1.2 tmp 0, nxt val
	// fill_out_hist_two (hist, ONE_ONE_TWO);

	// 1.1.3 tmp val, nxt 0
	// fill_out_hist_two (hist, ONE_ONE_THREE);

	// repeat for 3 histo
	fill_out_hist_two (hist, ONE_TWO_ONE);
	// fill_out_hist_two (hist, ONE_TWO_TWO);
	// fill_out_hist_two (hist, ONE_TWO_THREE);
	// fill_out_hist_two (hist, ONE_TWO_FOUR);
	// fill_out_hist_two (hist, ONE_TWO_FIVE);

	return hist;
}

// assumption: CHUNK_SIZE is 4. change hd0/1 appropriately.
static void fill_out_hist_two (const size_t c_size, token_tracking *hs, const int goal)
{
	token_tracking *tmp = hs;
	while (tmp) {
		switch (goal) {
			case ONE_ONE_ONE: {
// 				token hd0 [c_size] = {{2,22,"is",1},{3,23,"that",2},{2,24,"up",1},{5,25,"there",2}};
// 				unsigned int num0 = 0;
// 				token hd1 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
// 				unsigned int num1 = 0;
// 				token hd0 [c_size] = {{0,0,'\0',0},{4,23,"that",2},{2,24,"up",1},{5,25,"there",2}};
// 				unsigned int num0 = 1;
// 				token hd1 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
// 				unsigned int num1 = 1;
// 				token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{2,24,"up",1},{5,25,"there",2}};
// 				unsigned int num0 = 2;
// 				token hd1 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{0,0,'\0',0},{5,29,"spain",2}};
// 				unsigned int num1 = 2;
// 				token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{2,24,"up",1},{0,0,'\0',0}};
// 				unsigned int num0 = 3;
// 				token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{2,28,"in",1},{0,0,'\0',0}};
// 				unsigned int num1 = 3;
				token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				unsigned int num0 = 4;
				token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				unsigned int num1 = 4;

				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = num0;
				}
				else {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = num1;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (gb, tmp, c_size);
				break;
			}
			case ONE_ONE_TWO: {
				// token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd1 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				token hd0 [c_size] = {{3,{26,'t','h'},"the",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};
				token hd1 [c_size] = {{0,{0,'\0','\0'},'\0',0},{0,{'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{'\0','\0'},'\0',0}};
				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{0,0,'\0',0},{5,29,"spain",2}};
				// token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{2,28,"in",1},{0,0,'\0',0}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{2,28,"in",1},{0,0,'\0',0}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (gb, tmp, c_size);
				break;
			}
			case ONE_ONE_THREE: {
				// token hd0 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{3,26,"the",1}};
				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{0,0,'\0',0},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				token hd0 [c_size] = {{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{2,{28,'i','n'},"in",1},{0,{0,'\0','\0'},'\0',0}};
				token hd1 [c_size] = {{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{3,{26,'t','h'},"the",1}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (gb, tmp, c_size);
				break;
			}
			case ONE_TWO_ONE: {
				// 				token hd0 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// 				token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{2,28,"in",1},{3,26,"the",2}};
				// 				token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
// 				unsigned int num0 = 0;
// 				unsigned int num1 = 0;
// 				unsigned int num2 = 0;

				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{3,26,"the",2}};
				// token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				// unsigned int num0 = 1;
				// unsigned int num1 = 1;
				// unsigned int num2 = 1;

				token hd0 [c_size] = {{3,{26,'t','h'},"the",1},{0,{0,'\0','\0'},'\0',0},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};
				token hd1 [c_size] = {{0,{0,'\0','\0'},'\0',0},{6,{24,'m','a'},"mainly",2},{0,{0,'\0','\0'},'\0',0},{3,{26,'t','h'},"the",2}};
				token hd2 [c_size] = {{6,{23,'p','l'},"plains",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{0,{0,'\0','\0'},'\0',0}};
				unsigned int num0 = 1;
				unsigned int num1 = 2;
				unsigned int num2 = 1;

				// token hd0 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				// unsigned int num0 = 1;
				// unsigned int num1 = 2;
				// unsigned int num2 = 2;
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = num0;
				}
				else if (tmp -> prev_chunk && tmp -> next_chunk) {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = num1;
				}
				else {
					memcpy (tmp -> h_data, hd2, sizeof (token) * c_size);
					tmp -> h_num_free = num2;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (gb, tmp, c_size);
				break;
			}
			case ONE_TWO_TWO: {
				token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				token hd1 [c_size] = {{5,{25,'s','t'},"stays",1},{6,{24,'m','a'},"mainly",2},{2,{28,'i','n'},"in",1},{3,{26,'t','h'},"the",2}};
				token hd2 [c_size] = {{5,{23,'p','l'},"plains",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};

				// token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{3,26,"the",2}};
				// token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};

				// token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd1 [c_size] = {{0,0,'\0',0},{6,24,"mainly",2},{0,0,'\0',0},{3,26,"the",2}};
				// token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};

				// token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				else if (tmp -> prev_chunk && tmp -> next_chunk) {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else {
					memcpy (tmp -> h_data, hd2, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (tmp, c_size);
				break;
			}
			case ONE_TWO_THREE: {
				token hd0 [c_size] = {{3,{26,'t','h'},"the",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};
				token hd1 [c_size] = {{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0}};
				token hd2 [c_size] = {{5,{23,'p','l'},"plains",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};

				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};

				// token hd0 [c_size] = {{3,26,"the",1},{0,0,'\0',0},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{5,23,"plains",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};

				// token hd0 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				// token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else if (tmp -> prev_chunk && tmp -> next_chunk) {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				else {
					memcpy (tmp -> h_data, hd2, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (tmp, c_size);
				break;
			}
			case ONE_TWO_FOUR: {
				token hd0 [c_size] = {{3,{26,'t','h'},"the",1},{4,{27,'r','a'},"rain",2},{2,{28,'i','n'},"in",1},{5,{29,'s','p'},"spain",2}};
				token hd1 [c_size] = {{5,{25,'s','t'},"stays",1},{6,{24,'m','a'},"mainly",2},{2,{28,'i','n'},"in",1},{3,{26,'t','h'},"the",2}};
				token hd2 [c_size] = {{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0},{0,{0,'\0','\0'},'\0',0}};

				// token hd0 [c_size] = {{0,0,'\0',0},{4,27,"rain",2},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{3,26,"the",2}};
				// token hd2 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};

				// token hd0 [c_size] = {{3,26,"the",1},{0,0,'\0',0},{2,28,"in",1},{5,29,"spain",2}};
				// token hd1 [c_size] = {{0,0,'\0',0},{6,24,"mainly",2},{0,0,'\0',0},{3,26,"the",2}};
				// token hd2 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};

				// token hd0 [c_size] = {{3,26,"the",1},{4,27,"rain",2},{2,28,"in",1},{0,0,'\0',0}};
				// token hd1 [c_size] = {{5,25,"stays",1},{6,24,"mainly",2},{0,0,'\0',0},{0,0,'\0',0}};
				// token hd2 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else if (tmp -> prev_chunk && tmp -> next_chunk) {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 0;
				}
				else {
					memcpy (tmp -> h_data, hd2, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (tmp, c_size);
				break;
			}
			case ONE_TWO_FIVE: {
				token hd0 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				token hd1 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				token hd2 [c_size] = {{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0},{0,0,'\0',0}};
				if (!tmp -> prev_chunk) {
					memcpy (tmp -> h_data, hd0, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				else if (tmp -> prev_chunk && tmp -> next_chunk) {
					memcpy (tmp -> h_data, hd1, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				else {
					memcpy (tmp -> h_data, hd2, sizeof (token) * c_size);
					tmp -> h_num_free = 4;
				}
				gpuErrChk (cudaMemcpy (tmp -> d_data, tmp -> h_data, sizeof (token) * c_size, cudaMemcpyHostToDevice));
				print_token_tracking (tmp, c_size);
				break;
			}

			default:
				DBG_EXIT;
		}
		tmp = tmp -> next_chunk;
	}
}
***/