/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * histo_math.cu
 ******************************************************/

#include <cuda_milib.h>
#include <milib.h>

/*************  Imported Functions  **************/

/*************  Exported Functions  **************/
void math_histogram (all_bufs *ab);

/*************  Global variables  **************/
__device__ bool				d_Dlist_Sorted = false;

static bool get_Dlist_sort_status (void);
static bool get_Dlist_sort_status (void)
{
	bool h_srt = false;
	gpuErrChk (cudaMemcpyFromSymbol (&h_srt, d_Dlist_Sorted, sizeof (bool)));

	return h_srt;
}

static void set_Dlist_sort_true (void);
static void set_Dlist_sort_true (void)
{
	bool h_srt = true;
	bool *d_ads = NULL;

	gpuErrChk (cudaGetSymbolAddress ((void **) &d_ads, d_Dlist_Sorted));
	gpuErrChk (cudaMemcpy (d_ads, &h_srt, sizeof (bool), cudaMemcpyHostToDevice));
}

static void set_Dlist_sort_false (void);
static void set_Dlist_sort_false (void)
{
	bool h_srt = false;
	bool *d_ads = NULL;

	gpuErrChk (cudaGetSymbolAddress ((void **) &d_ads, d_Dlist_Sorted));
	gpuErrChk (cudaMemcpy (d_ads, &h_srt, sizeof (bool), cudaMemcpyHostToDevice));
}

static void __global__ K_Histo_Math_reduce_to_one_val (uint32_t *d_tval, unsigned int half);
static void __global__ K_Histo_Math_reduce_to_one_val (uint32_t *d_tval, unsigned int half)
{
	const unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= half) return;
	const unsigned int other_side = half + gTh;
	// CudaDbgPrn ("th: %u: d_tval %u, other_side %u", (unsigned) gTh, (unsigned) d_tval [gTh], (unsigned) other_side);

	d_tval [gTh] += d_tval [other_side];
	d_tval [other_side] = '\0';
}

static void __global__ K_Histo_Math_intermed_variance (uint32_t *d_proc, const unsigned int len, unsigned int h_avg);
static void __global__ K_Histo_Math_intermed_variance (uint32_t *d_proc, const unsigned int len, unsigned int h_avg)
{
	const unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	if (gTh >= len) return;

	int tmp = (int) (d_proc [gTh] - h_avg);
	d_proc [gTh] = (uint32_t) (tmp * tmp);
	// CudaDbgPrn ("th: %u: d_tval %u, tmp %u", (unsigned) gTh, (unsigned) d_proc [gTh], (unsigned) tmp);
}

void __global__ K_Histo_Math_cp_nums_to_dlist (words *num_list, token **tok, unsigned int *math_loc, unsigned int tok_start, const size_t c_size);
void __global__ K_Histo_Math_cp_nums_to_dlist (words *num_list, token **tok, unsigned int *math_loc, unsigned int tok_start, const size_t c_size)
{
	const unsigned int gTh = blockIdx.x * blockDim.x + threadIdx.x;
	tok_start += gTh;
	if (tok_start >= c_size) return;
	assert (tok [tok_start]);

	unsigned int curr_math = atomicAdd (math_loc, (unsigned int) 1);
	__threadfence ();
	cuda_MemCpy (num_list + curr_math, &(tok [tok_start] -> wd), sizeof (words));

	// CudaDbgPrn ("th: %u: curr_math %u, num_list: %.20s num %u", (unsigned) gTh, (unsigned) curr_math, num_list [curr_math].str, (unsigned) num_list [curr_math].num);
}

static __device__ inline void Dlist_swap (words *curr, words *next);
static __device__ inline void Dlist_swap (words *curr, words *next)
{
	uint32_t		t_num;
	unsigned int	t_len;
	unsigned char	t_str [STAT_MAX_WORD_SIZE];

	// curr -> temp
	t_num	= (uint32_t) curr -> num;
	t_len	= (unsigned int) curr -> len;
	cuda_strn_cpy (t_str, curr -> str, (unsigned) t_len);
	t_str [t_len] = (unsigned char) '\0';

	// next -> curr
	curr -> num		= (uint32_t) next -> num;
	curr -> len		= (unsigned int) next -> len;
	cuda_strn_cpy (curr -> str, next -> str, (unsigned) curr -> len);
	curr -> str [curr -> len] = (unsigned char) '\0';

	// temp -> next
	next -> num		= (uint32_t) t_num;
	next -> len		= (unsigned int) t_len;
	cuda_strn_cpy (next -> str, t_str, (unsigned) t_len);
	next -> str [t_len] = (unsigned char) '\0';
}

static void __global__ K_Histo_Math_Sort_Dlist_Even (words *curr, const unsigned int Dlist_len);
static void __global__ K_Histo_Math_Sort_Dlist_Even (words *curr, const unsigned int Dlist_len)
{
	const unsigned int gTh = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	if (gTh >= Dlist_len - 1) return;

	// CudaDbgPrn ("Estart: th %u|curr: %.10s nm %u|nxt: %.10s nm %u", (unsigned) gTh, curr [gTh].str, (unsigned) curr [gTh].num, curr [gTh + 1].str, (unsigned) curr [gTh + 1].num);

	if (curr [gTh].num > curr [gTh + 1].num) {
		Dlist_swap (&curr [gTh], &curr [gTh + 1]);
		d_Dlist_Sorted = false;
		// CudaDbgPrn ("%u, E: Dlist Swapped", (unsigned) gTh);
	}

	// CudaDbgPrn ("Oend: th%u|curr: %.10s nm %u|nxt: %.10s nm %u", (unsigned) gTh, curr [gTh].str, (unsigned) curr [gTh].num, curr [gTh + 1].str, (unsigned) curr [gTh + 1].num);
}

static void __global__ K_Histo_Math_Sort_Dlist_Odd (words *curr, const unsigned int Dlist_len);
static void __global__ K_Histo_Math_Sort_Dlist_Odd (words *curr, const unsigned int Dlist_len)
{
	const unsigned int gTh = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + 1;
	if (gTh >= Dlist_len) return;

	// CudaDbgPrn ("Ostart: th %u|curr: %.10s nm %u|nxt: %.10s nm %u", (unsigned) gTh, curr [gTh].str, (unsigned) curr [gTh].num, curr [gTh + 1].str, (unsigned) curr [gTh + 1].num);

	if (curr [gTh].num > curr [gTh + 1].num) {
		Dlist_swap (&curr [gTh], &curr [gTh + 1]);
		d_Dlist_Sorted = false;
		// CudaDbgPrn ("%u, O: Dlist Swapped", (unsigned) gTh);
	}

	// CudaDbgPrn ("Oend: th%u|curr: %.10s nm %u|nxt: %.10s nm %u", (unsigned) gTh, curr [gTh].str, (unsigned) curr [gTh].num, curr [gTh + 1].str, (unsigned) curr [gTh + 1].num);
}

static void histo_math_cp_sort_Dlist (all_bufs *ab, math_results *hm);
static void histo_math_cp_sort_Dlist (all_bufs *ab, math_results *hm)
{
	unsigned int i = 0;
	{
		token_tracking *tmp;
		for (tmp = ab -> h_ttrack; tmp; tmp = tmp -> next) i++;

		unsigned int *d_math_loc = NULL;
		gpuErrChk (cudaMalloc (&d_math_loc, sizeof (unsigned int)));
		gpuErrChk (cudaMemset (d_math_loc, 0x00, sizeof (unsigned int)));

		dim3 block (ab -> h_ttrack -> chunk_size);
		dim3 grid ((i * ab -> h_ttrack -> chunk_size + block.x - 1) / block.x);

		for (i = 0, tmp = ab -> h_ttrack; tmp; tmp = tmp -> next, i++) {
			K_Histo_Math_cp_nums_to_dlist <<<grid, block>>> (hm -> d_num_list, tmp -> d_data, d_math_loc, tmp -> h_num_free, tmp -> chunk_size);
			CUDA_GLOBAL_CHECK;
		}
		CUDA_FREE (d_math_loc);
	}

	dim3 grid ((i * ab -> h_ttrack -> chunk_size + ab -> def_gpu_calc.block.x - 1) / ab -> def_gpu_calc.block.x);
	set_Dlist_sort_false ();
	while (false == get_Dlist_sort_status ()) {
		set_Dlist_sort_true ();
		K_Histo_Math_Sort_Dlist_Even <<<grid, ab -> def_gpu_calc.block>>> (hm -> d_num_list, hm -> list_len);
		CUDA_GLOBAL_CHECK;
		K_Histo_Math_Sort_Dlist_Odd <<<grid, ab -> def_gpu_calc.block>>> (hm -> d_num_list, hm -> list_len);
		CUDA_GLOBAL_CHECK;
	}

	gpuErrChk (cudaMemcpy (hm -> h_num_list, hm -> d_num_list, sizeof (words) * hm -> list_len, cudaMemcpyDeviceToHost));

	/***
	Dbg ("=== %u odd/even pass done ===", i++);
	for (i = 0; i < hm -> list_len; i++) {	// RS_DEBUG
		Dbg ("%u:%.20s len %u num %u ", i, hm -> h_num_list [i].str, hm -> h_num_list [i].len, hm -> h_num_list [i].num);
	}
	***/
}

static void __global__ K_Histo_cp_to_proc_list (uint32_t *d_proc, words *d_num, const unsigned int len);
static void __global__ K_Histo_cp_to_proc_list (uint32_t *d_proc, words *d_num, const unsigned int len)
{
	const unsigned int gTh = (blockIdx.x * blockDim.x + threadIdx.x);
	if (gTh >= len) return;

	d_proc [gTh] = d_num [gTh].num;
}

static void histo_math_basic_calc (all_bufs *ab, math_results *hm);
static void histo_math_basic_calc (all_bufs *ab, math_results *hm)
{
	hm -> avg.num = (double) hm -> h_tot_wds_pst_sw / hm -> h_tot_unique_wds;

	unsigned int half = hm -> h_tot_unique_wds / 2;
	unsigned int	median, val, i;	// Median Assumption: CHUNK_SIZE even.
	for (i = 0; 0 == hm -> h_num_list [i].num && i < hm -> list_len; i++);
	if ((i + half) > hm -> list_len) {
		Dbg ("i %u, half %u, hm->list_len %u", i, half, hm -> list_len);
		goto error;
	}
	median = hm -> h_num_list [i + half - 1].num;
	median += hm -> h_num_list [i + half].num;
	hm -> median.num = (double) median / 2;

	val = (unsigned int) hm -> median.num;
	for (i = 0; val > hm -> h_num_list [i].num && i < hm -> list_len; i++);
	Dbg ("Median: loc %u, val %u", i, hm -> h_num_list [i].num);
	memcpy (&(hm -> median), &(hm -> h_num_list [i]), sizeof (words));

	val = (unsigned int) hm -> avg.num;
	for (i = 0; val > hm -> h_num_list [i].num && i < hm -> list_len; i++);
	Dbg ("average: loc %u val %u", i, hm -> h_num_list [i].num);
	memcpy (&(hm -> avg), &(hm -> h_num_list [i]), sizeof (words));

	for (i = 0; 0 == hm -> h_num_list [i].num && i < hm -> list_len; i++);
	memcpy (&(hm -> min), &(hm -> h_num_list [i]), sizeof (words));
	Dbg ("Min loc %u val %u", i, hm -> min.num);

	memcpy (&(hm -> max), &(hm -> h_num_list [hm -> list_len - 1]), sizeof (words));
	Dbg ("Max loc %u val %u", hm -> list_len - 1, hm -> max.num);
	return;

error:
	Dbg ("Invalid state");
	exit (1);
}

static void histo_math_std_dev (all_bufs *ab, math_results *hm);
static void histo_math_std_dev (all_bufs *ab, math_results *hm)
{
	// dim3	block (ab -> gc -> dev_prop -> maxThreadsPerBlock / 4); - test for these
	// dim3	block (ab -> gc.dev_prop.maxThreadsPerBlock / 2);
	dim3	block (ab -> gc.dev_prop.maxThreadsPerBlock);
	dim3	grid	((hm -> list_len + block.x - 1) / block.x);

	uint32_t	*d_plist = NULL;	// list for reduction
	gpuErrChk (cudaMalloc (&d_plist, sizeof (uint32_t) * hm -> list_len));
	gpuErrChk (cudaMemset (d_plist, '\0', sizeof (uint32_t) * hm -> list_len));

	K_Histo_cp_to_proc_list <<<grid, block>>>  (d_plist, hm -> d_num_list, hm -> list_len);
	CUDA_GLOBAL_CHECK;

	K_Histo_Math_intermed_variance <<<grid, block>>> (d_plist, hm -> list_len, hm -> avg.num);
	CUDA_GLOBAL_CHECK;

	// dim3 grid_new	(((hm -> list_len/2) + block.x - 1) / block.x);
	grid = ((hm -> list_len/2) + block.x - 1) / block.x;
	for (unsigned int half = hm -> list_len/2; half; half /= 2) {
		K_Histo_Math_reduce_to_one_val <<<grid, block>>> (d_plist, half);
		CUDA_GLOBAL_CHECK;
	}

	uint32_t h_std_dev;
	gpuErrChk (cudaMemcpy (&h_std_dev, d_plist, sizeof (unsigned int), cudaMemcpyDeviceToHost));
	hm -> std_dev = (double) sqrt (h_std_dev / hm -> h_tot_unique_wds);

	CUDA_FREE (d_plist);
}

static void math_histogram_setup (math_results *hm, token_tracking *hist, unsigned int tot_hist);
static void math_histogram_setup (math_results *hm, token_tracking *hist, unsigned int tot_hist)
{
	// set list_len & the list of values
	if ((tot_hist * hist -> chunk_size) > hm -> list_len) {
		hm -> list_len = tot_hist * hist -> chunk_size;
		CUDA_FREE (hm -> d_num_list);
		gpuErrChk (cudaMalloc (&hm -> d_num_list, sizeof (words) * hm -> list_len));

		FREE (hm -> h_num_list);
		hm -> h_num_list = (words *) malloc (sizeof (words) * hm -> list_len);
		check_mem (hm -> h_num_list);
	}
	gpuErrChk (cudaMemset (hm -> d_num_list, '\0', sizeof (words) * hm -> list_len));
	memset (hm -> h_num_list, '\0', sizeof (words) * hm -> list_len);

	return;

error:
	Dbg ("Out of memory");
	exit (1);
}

// do & get frequency- statistical analysis
void math_histogram (all_bufs *ab)
{
	unsigned int tot_hist = 0;
	ab -> h_math.h_tot_unique_wds = ab -> h_math.h_tot_wds_pst_sw = 0;
	for (token_tracking *tmp = ab -> h_ttrack; tmp; tmp = tmp -> next) {
		tot_hist++;
		ab -> h_math.h_tot_unique_wds += (tmp -> chunk_size - tmp -> h_num_free);
		for (unsigned int i = 0; i < tmp -> chunk_size; i++) {
			ab -> h_math.h_tot_wds_pst_sw += tmp -> h_data [i].wd.num;
		}
	}
	gpuErrChk (cudaMemcpy (&(ab -> h_math.h_tot_wds_pre_sw), ab -> d_wds, sizeof (unsigned int), cudaMemcpyDeviceToHost));

	math_histogram_setup (&ab -> h_math, ab -> h_ttrack, tot_hist);
	histo_math_cp_sort_Dlist (ab, &ab -> h_math);
	histo_math_basic_calc (ab, &ab -> h_math);
	histo_math_std_dev (ab, &ab -> h_math);
}
