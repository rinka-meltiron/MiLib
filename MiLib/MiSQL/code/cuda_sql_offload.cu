/*********************************
 * These are all the cuda related code and
 * the main function: GetResultofSQLQueryfromGPU
 *
 * The rest of the GPU related host functions are
 * in gpu_prepare_sql.cpp
 *
 * Rinka Singh - Melt Iron
 *********************************/

#include <sys/time.h>
#include "../include/cuda_milib.h"
#include "sql_query.hpp"

extern mi_sqltable *sql_fetch_result (mi_sqlres *res, const uint64_cu res_rows, bool new_query);
extern "C" gpu_config *setup_gpu_card (void);
extern "C" void add_and_cp_to_buffer (mi_sqlres *mi_res, MYSQL_ROW row, mi_sqltable *hd_local_row, uint64_cu num_row);

static uint64_cu cuda_process_answer (connection_details *sqld, gpu_config *cfg);
static uint64_cu compress_and_extract_result (connection_details *sqld, mi_params_for_kernel1 *d_p, gpu_config *cfg, struct timeval *t0Diff);
static void load_params_into_card1 (connection_details *sqld, mi_sql *query);
static void load_params_into_card2 (connection_details *cd);
static void load_str_tobe_matched_into_card (mi_sql *query);
static void pre_gpu_query_setup (connection_details *sqld);
static void post_gpu_query_setup (connection_details *sqld);

extern "C" mi_sqltable *create_cuda_buffer (mi_sqlres *res);
extern "C" void GetResultofSQLQueryfromGPU (MYSQL *conn, connection_details *mysqlD);

static __global__ void K_dummy (void);
static __global__ void K_pattern_match (mi_params_for_kernel1 *mp);
static __global__ void K_compress_result (mi_params_for_kernel1 *mp, uint64_cu *d_c_res);

// void test_table (uint64_cu r_count, mi_sqlres *res, gpu_config *cfg);
// static __global__ void K_test_table (mi_sqltable *tab);

// Variables shared with the device
// constant variable declarations
__constant__ uint64_cu		d_mrow [CONST_PARAMS2];
#define CONST_MROW		d_mrow [CONST_PARAMS0]	// mrow - max rows
#define MAX_IDX			d_mrow [CONST_PARAMS1]		// CONST_MROW * CONST_FCNT

__constant__ unsigned int cMt [CONST_PARAMS4];
#define CONST_MT		cMt [CONST_PARAMS0]		// mt - max threads
#define CONST_FID		cMt [CONST_PARAMS1]		// fid - field id
#define CONST_TLEN		cMt [CONST_PARAMS2]		// tlen - strlen of pat
#define CONST_FCNT		cMt [CONST_PARAMS3]		// fcnt - number of fields
// the four cells are mt, fid, tlen, fcnt;

__constant__ unsigned char	d_patt [DB_NAME];

// K_compress_result - 0 - no result
__device__ uint64_cu roc [CONST_PARAMS1] = {0};
#define result_on_card		roc [CONST_PARAMS0]	// result_on_card location
__device__ volatile unsigned long long int id [CONST_PARAMS1] = {0};
__device__ unsigned int locks [CONST_PARAMS1] = {0};

gpu_config *setup_gpu_card (void)
{
	unsigned i = 0;

	gpuErrChk (cudaDeviceReset ());
	CUdevice	cud;
	CUresult	cres = cuDeviceGet (&cud, 0);
	CUcontext	cuctx = 0;

	gpu_config *gc = (gpu_config *) malloc (sizeof (gpu_config));
	check_mem (gc);
	if (CUDA_SUCCESS != (i = cuCtxCreate (&cuctx, CU_CTX_SCHED_AUTO, cud))) {
		Dbg ("cuCtxCreate returned %u", i);
	}
	gc -> cuc = cuctx;

	gpuErrChk (cudaGetDeviceCount (&gc -> dev_count));
	// Dbg("Device count %d", gc -> dev_count);

	gc -> dev_prop = (cudaDeviceProp *) malloc (sizeof (cudaDeviceProp));
	check_mem (gc -> dev_prop);

	gpuErrChk (cudaGetDeviceProperties (gc -> dev_prop, 0));

	Dbg ("Dev prop name: %s, sharedMemPerBlock %u warpSize %u maxThreadsPerBlock %u, maxthreads per mprocessor %d",
		gc -> dev_prop -> name , (unsigned) gc -> dev_prop -> sharedMemPerBlock, gc -> dev_prop -> warpSize, gc -> dev_prop -> maxThreadsPerBlock, gc -> dev_prop -> maxThreadsPerMultiProcessor);

	return gc;

error:
	return FAILED;
}

/*******************************************************************/
static uint64_cu cuda_process_answer (connection_details *sqld, gpu_config *cfg)
{
	struct timeval	t1, t2, tvDiff, bt;
	unsigned int th = cfg -> dev_prop -> maxThreadsPerBlock / 2;
	dim3 block (th);
	dim3 grid ((sqld -> res -> row_count + block.x - 1)/block.x);

	load_params_into_card1 (sqld, sqld -> sql);
	load_params_into_card2 (sqld);

	// test_table (sqld -> res -> row_count, sqld -> res, sqld -> cfg);

	gettimeofday (&t1, NULL);
	K_pattern_match <<<grid, block>>> (sqld -> d_p);
	gpuErrChk (cudaPeekAtLastError ());
	gpuErrChk (cudaDeviceSynchronize());
	gettimeofday (&t2, NULL);

	if (sqld -> time) {
		timeval_subtract (&tvDiff, &t2, &t1);
		printf ("CUDA: Time taken by query %ld.%06ld sec\n", tvDiff.tv_sec, tvDiff.tv_usec);
		bt.tv_sec = tvDiff.tv_sec;
		bt.tv_usec = tvDiff.tv_usec;
	}

	// we know how many records were found result_on_card has the number
	return compress_and_extract_result (sqld, sqld -> d_p, cfg, &bt);
}

static uint64_cu compress_and_extract_result (connection_details *sqld, mi_params_for_kernel1 *d_p, gpu_config *cfg, struct timeval *t0Diff)
{
	struct timeval	t1, t3, tvDiff;
	uint64_cu h_result_on_card [CONST_PARAMS1];
	gpuErrChk (cudaMemcpyFromSymbol (h_result_on_card, roc [CONST_PARAMS0], sizeof (uint64_cu)));	// result_on_card
	// Dbg ("result_on_card %llu", *h_result_on_card);

	dim3 block (cfg -> dev_prop -> maxThreadsPerBlock / 2);
	// dim3 block (1024);	// from calculator
	dim3 grid ((sqld -> res -> row_count + block.x - 1)/block.x);

	gettimeofday (&t1, NULL);
	if (0 != *h_result_on_card) {
		gpuErrChk (cudaMalloc (&(sqld -> res -> d_c_res), ((uint64_cu) *h_result_on_card) * sizeof (uint64_cu)));
		gpuErrChk (cudaMemset (sqld -> res -> d_c_res, (uint64_cu) 0, *h_result_on_card * sizeof (uint64_cu)));
		sqld -> res -> h_c_res = (uint64_cu *) calloc (*h_result_on_card * sizeof (uint64_cu), sizeof (uint64_cu));
		check_mem (sqld -> res -> h_c_res);

		K_compress_result <<<grid, block>>> (d_p, sqld -> res -> d_c_res);	// result_on_card updated here
		gpuErrChk (cudaPeekAtLastError ());
		gpuErrChk (cudaDeviceSynchronize());

		gpuErrChk (cudaMemcpy (sqld -> res -> h_c_res, sqld -> res -> d_c_res, ((uint64_cu) *h_result_on_card) * sizeof (uint64_cu), cudaMemcpyDeviceToHost));
	}
	else {
		sqld -> res -> d_c_res = NULL;
		sqld -> res -> h_c_res = NULL;
	}
	gettimeofday (&t3, NULL);

	/***
	for (uint64_cu i = 0; i < *h_result_on_card; i++) {
		Dbg ("after cpy sqld...->h_c_res [%llu] = %llu", i, sqld -> res -> h_c_res [i]);
	}
	***/

	if (sqld -> time) {
		timeval_subtract (&tvDiff, &t3, &t1);
		printf ("CUDA: Time taken to fetch compressed results %ld.%06ld sec & Tot time %ld.%06ld\n", tvDiff.tv_sec, tvDiff.tv_usec, t0Diff -> tv_sec + tvDiff.tv_sec, t0Diff -> tv_usec + tvDiff.tv_usec);
	}

	return *h_result_on_card;

error:
	return 0;
}

static void load_params_into_card1 (connection_details *sqld, mi_sql *query)
{
	// allocate global variables cMt, d_patt, roc
	uint64_cu h_row [CONST_PARAMS2];
	h_row [CONST_PARAMS0] = (uint64_cu) sqld -> res -> row_count;
	h_row [CONST_PARAMS1] = (uint64_cu) sqld -> res -> row_count * sqld -> res -> field_count;
	gpuErrChk (cudaMemcpyToSymbol (d_mrow, h_row, sizeof (uint64_cu) * CONST_PARAMS2));
}

static void load_str_tobe_matched_into_card (mi_sql *query)
{
	unsigned char h_pat [DB_NAME];
	strncpy ((char *) h_pat, (const char *) query -> str_tobe_matched, query -> str_len);
	gpuErrChk (cudaMemcpyToSymbol (d_patt, h_pat, sizeof (unsigned char) * query -> str_len));
}

static void load_params_into_card2 (connection_details *cd)
{
	mi_params_for_kernel1	hd_pk;
	// pass the params to the kernel
	hd_pk.d_table	= cd -> res -> d_table;
	hd_pk.result	= cd -> res -> d_res;

	gpuErrChk (cudaMalloc (&(cd -> d_p), sizeof (mi_params_for_kernel1)));
	gpuErrChk (cudaMemcpy (cd -> d_p, &hd_pk, sizeof (mi_params_for_kernel1), cudaMemcpyHostToDevice));
}

/**************
 * we get called from cuda_process_answer - goal is to compress & get everything down
 * if number of results is >40% - just do a cudaMemcpy.  This will be done in the kernel
 * if less than 50%
 * 		while (result_on_card != 0)
 * 			cudaMemcpy data from card to host.
 * 			hand host_data to add_to_h_res
 * 		endwhile
 * 		return the full data received to cuda_process_answer
 * 		cuda_process_answer will store data for sql_fetch result
 **************/
mi_sqltable *create_cuda_buffer (mi_sqlres *res)
{
	mi_sqltable		*hd_local_row = (mi_sqltable *) malloc (sizeof (mi_sqltable));

	// table and data on host
	res -> h_table = (mi_sqltable *) malloc (res -> row_count * res -> field_count * sizeof (mi_sqltable));
	check_mem (res -> h_table);
	res -> h_data = (unsigned char *) malloc (res -> db_bytes * sizeof (unsigned char));
	check_mem (res -> h_data);

	// data on host
	// data & table on card
	gpuErrChk (cudaMalloc (&(res -> d_table), (res -> row_count + 1) * res -> field_count * sizeof (mi_sqltable)));
	gpuErrChk (cudaMalloc (&(res -> d_data), res -> db_bytes * sizeof (unsigned char)));
	gpuErrChk (cudaMemset (res -> d_data, '\0', res -> db_bytes * sizeof (unsigned char)));

	// result buffer on d
	gpuErrChk (cudaMalloc (&res -> d_res, ((res -> row_count + 1) * sizeof (uint64_cu))));
	gpuErrChk (cudaMemset (res -> d_res, 0, ((res -> row_count + 1) * sizeof (uint64_cu))));

	// size_t		free, total;
	// cuMemGetInfo (&free, &total);

	return hd_local_row;

error:
	Dbg ("Memory not allocated");
	return FAILED;
}

// doing this for each row
void add_and_cp_to_buffer (mi_sqlres *mi_res, MYSQL_ROW row, mi_sqltable *hd_local_row, uint64_cu num_row)
{
	static unsigned char	*tmp_d_data = mi_res -> d_data;	// table data
	static mi_sqltable		*tmp_d_table = mi_res -> d_table;
	static mi_sqltable		*tmp_h_table = mi_res -> h_table;
	static unsigned char	*tmp_h_data = mi_res -> h_data;
	static int				j = 0;
	// static uint64_cu		tot = 0;

	// frm mysql table: update our table
	for (uint32_cu i = 0; i < mi_res -> field_count; i++) {
		uint32_cu l = strlen (row [i]);		// get len

		strncpy ((char *) tmp_h_data, row [i], l);	// copy to host
		// Dbg ("host:%u fld %u row: %s", (unsigned) num_row, (unsigned) i, row [i]);
		tmp_h_data [l] = '\0';

		tmp_h_table -> length = l;
		tmp_h_table -> row = num_row;
		tmp_h_table -> fld = i;
		tmp_h_table -> element = tmp_h_data;

		// tot += l + 1;
		// Dbg ("tot len for tmp_d_data %u against max: %u", (unsigned) tot, (unsigned) mi_res -> db_bytes);
		gpuErrChk (cudaMemcpy (tmp_d_data, row [i], (l * sizeof (unsigned char)), cudaMemcpyHostToDevice));
		// Dbg ("cp'd %s to d_data @ ptr: %p", row [i], tmp_d_data);

		hd_local_row -> element = tmp_d_data;
		hd_local_row -> length = l;
		hd_local_row -> row = num_row;
		hd_local_row -> fld = i;
		gpuErrChk (cudaMemcpy (tmp_d_table, hd_local_row, sizeof (mi_sqltable), cudaMemcpyHostToDevice));

		tmp_h_table++;
		tmp_d_table++;
		tmp_h_data += (l + 1); // mv to next loc
		tmp_d_data += (l + 1); // mv to next loc

		if (j++ >= 100000) {
			printf (".");
			j = 0;
		}
		// Dbg ("hd_local_row[%d]: %p", (int) i, tmp_d_data);
		// RS: we are not checking for buffer overruns...
	}

	// Dbg ("Row %u", (unsigned) num_row);
}

static void pre_gpu_query_setup (connection_details *sqld)
{
	static bool first_time = false;

	load_str_tobe_matched_into_card (sqld -> sql);

	// load constant variables
	unsigned int h_mt [CONST_PARAMS4];
	h_mt [CONST_PARAMS0] = (unsigned int) (sizeof (unsigned) * sqld -> res -> row_count * sqld -> sql -> str_len);
	h_mt [CONST_PARAMS1] = (unsigned int) sqld -> res -> field_id;
	h_mt [CONST_PARAMS2] = (unsigned int) sqld -> sql -> str_len;
	h_mt [CONST_PARAMS3] = (unsigned int) sqld -> res -> field_count;
	// Dbg ("before mt %u, field id %u, plen %u fcnt %u", (unsigned) h_mt [CONST_PARAMS0], (unsigned) h_mt [CONST_PARAMS1], (unsigned) h_mt [CONST_PARAMS2], (unsigned) h_mt [CONST_PARAMS3]);
	gpuErrChk (cudaMemcpyToSymbol (cMt, h_mt, sizeof (unsigned int) * CONST_PARAMS4));

	if (!first_time) {
		// warmup needed to make sure driver initialization is done.
		K_dummy <<<1, 1>>> ();
		gpuErrChk (cudaPeekAtLastError ());
		gpuErrChk (cudaDeviceSynchronize());

		first_time = true;
	}
}

static void post_gpu_query_setup (connection_details *sqld)
{
	// zero out d_res
	gpuErrChk (cudaMemset (sqld -> res -> d_res, 0, ((sqld -> res -> row_count + 1) * sizeof (uint64_cu))));
	CUDA_FREE (sqld -> res -> d_c_res);
	FREE (sqld -> res -> h_c_res);

	// reset global variables
	// load constant variables
	unsigned int h_mt [CONST_PARAMS4];
	h_mt [CONST_PARAMS0] = (unsigned int) (sizeof (unsigned) * sqld -> res -> row_count * sqld -> sql -> str_len);
	h_mt [CONST_PARAMS1] = (unsigned int) 0;
	h_mt [CONST_PARAMS2] = (unsigned int) 0;
	h_mt [CONST_PARAMS3] = (unsigned int) sqld -> res -> field_count;
	// Dbg ("before mt %u, field id %u, plen %u fcnt %u", (unsigned) h_mt [CONST_PARAMS0], (unsigned) h_mt [CONST_PARAMS1], (unsigned) h_mt [CONST_PARAMS2], (unsigned) h_mt [CONST_PARAMS3]);
	gpuErrChk (cudaMemcpyToSymbol (cMt, h_mt, sizeof (unsigned int) * CONST_PARAMS4));

	uint64_cu h_result_on_card [CONST_PARAMS1] = {0};
	gpuErrChk (cudaMemcpyToSymbol (roc [CONST_PARAMS0], h_result_on_card, sizeof (uint64_cu)));	// result_on_card

	unsigned int h_locks [CONST_PARAMS1] = {0};
	gpuErrChk (cudaMemcpyToSymbol (locks, h_locks, sizeof (unsigned int)));	// locks
}

void GetResultofSQLQueryfromGPU (MYSQL *conn, connection_details *mysqlD)
{
	mi_sqltable	*out_row;

	pre_gpu_query_setup (mysqlD);

	uint64_cu res_card = cuda_process_answer (mysqlD, mysqlD -> cfg);
	if (0 == res_card) {
		printf ("No rows found\n");
		return;
	}

	bool n_query = true;
	while ((out_row = sql_fetch_result (mysqlD -> res, res_card, n_query)) != NULL) {
		for (uint32_cu i = 0; i < mysqlD -> res -> field_count; i++) {
			printf ("%s, ", out_row[i].element);
		}
		printf ("\n");
		n_query = false;
	}
	post_gpu_query_setup (mysqlD);
}

static __global__ void K_dummy (void)
{
	unsigned int i = CONST_MT;
	i = CONST_FID * CONST_TLEN;
	i = CONST_FCNT * MAX_IDX;

	i = (unsigned int) CONST_MROW;

	i = (unsigned int) roc [CONST_PARAMS0]; // result_on_card
	i++;
}

/************ This doesn't work yet *************************
static __global__ void K_pattern_match (mi_params_for_kernel1 *mp)
{
	uint32_cu gbl_th = blockIdx.x * blockDim.x + threadIdx.x;
	 * constant declarations at the top of the file
	// C_TOT_TH_NEEDED	cMT [0] total threads in block: max fields if cuda_urnrolled > 1
	// CONST_FID		cMT [1] field_id
	// CONST_TLEN		cMT [2] strlen of pattern
	// CONST_FCNT		cMt [3]	number of fields
	// C_TOT_TH_CHUNKS	th chunks of TLEN
	// CONST_MROW		max_rows
	// CONST_TOT_FLD	tot fields (row x flds) in table
	extern __shared__ unsigned int sh_res [];
	uint64_cu rec_id		= (uint64_cu) gbl_th / CONST_TLEN;	// 0 - n
	if (rec_id >= CONST_MROW) return;

	unsigned int pat_idx	= (unsigned int) gbl_th % CONST_TLEN;	// 0 - pat_len
	uint64_cu rec_idx		= (uint64_cu) rec_id * CONST_FCNT + CONST_FID; // record id pointing to the specific db field

	sh_res [gbl_th] = (unsigned int) 0;	// initialize
	// CudaDbgPrn ("rec_id %u, gbl_th %u, FCNT %u FID %u, TLEN %u", (unsigned) rec_id, (unsigned) gbl_th, (unsigned) CONST_FCNT, (unsigned) CONST_FID, (unsigned) CONST_TLEN);
	__syncthreads ();

	if (CONST_TLEN != mp -> d_table [rec_idx].length) {
		CudaDbgPrn ("th: %u wrong length %u, rec_idx %u %s", (unsigned) gbl_th, (unsigned) mp -> d_table [rec_idx].length, rec_idx, mp -> d_table [rec_idx].element);
		return;
	}

	CudaDbgPrn ("th:%u, row %u, fld %u index of elem %u, patt %s, elem %s", (unsigned int) gbl_th, (unsigned int) mp -> d_table [rec_idx].row, (unsigned int) mp -> d_table [rec_idx].fld, pat_idx, d_patt, mp -> d_table [rec_idx].element);

	if (mp -> d_table [rec_idx].element [pat_idx] == d_patt [pat_idx]) {
		CudaDbgPrn ("th %d, d_table [%u] %c & d_patt %c is true", (unsigned int) gbl_th, (unsigned int) pat_idx, mp -> d_table [rec_idx].element [pat_idx], d_patt [pat_idx]);

		sh_res [gbl_th] = (unsigned int) 1;	// char match succeeded
	}
	else {	// delete the else after debugging
		CudaDbgPrn ("th %d, d_table [%u] %s & d_patt %s is false", (unsigned int) gbl_th, (unsigned int) pat_idx, mp -> d_table [rec_idx].element, d_patt);
	}

	if (0 == pat_idx) {
		mp -> result [rec_id] = (unsigned int) sh_res [gbl_th];
	}
	else {
		atomicMin ((int *) &(mp -> result [rec_id]), (int) sh_res [gbl_th]);
		CudaDbgPrn ("th %u: result [%u] = %d", gbl_th, rec_id, mp -> result [rec_id]);
	}

	if (gbl_th >= CONST_MROW) return;

	if (1 == mp -> result [rec_id] && 0 == pat_idx) {
		atomicAdd ((int *) &result_on_card, (int) 1);
		__threadfence ();
		CudaDbgPrn ("inside true row %u result_on_card %d,th %u rec_id %d", (unsigned int) mp -> d_table [rec_id].row, (unsigned int) result_on_card, (unsigned int) gbl_th, (int) rec_id);
	}
}
*********************************/

static __global__ void K_pattern_match (mi_params_for_kernel1 *mp)
{
	register uint32_cu i;
	uint64_cu gbl_th = (uint64_cu) (blockIdx.x * blockDim.x + threadIdx.x);
	if (gbl_th >= CONST_MT) return;

	// Performance: TBD: http://stackoverflow.com/questions/29487568/beginner-help-on-cuda-code-performance?rq=1

	register uint64_cu	idx = CONST_FCNT * gbl_th + CONST_FID;
	__syncthreads ();

	// CudaDbgPrn ("idx %u, MAX_IDX %u", (unsigned) idx, (unsigned) MAX_IDX);
	if (idx >= MAX_IDX) return;
	// CudaDbgPrn ("idx %u, CONST_FCNT %u, gbl_th %u CONST_FID %u CONST_TLEN %u", (unsigned) idx, (unsigned) CONST_FCNT, (unsigned) gbl_th, (unsigned) CONST_FID, (unsigned) CONST_TLEN);
	// CudaDbgPrn ("before comparing row len %u %u, str %s", (unsigned int) mp -> d_table [idx].row, mp -> d_table [idx].length, mp -> d_table [idx].element);
	if (mp -> d_table [idx].length != CONST_TLEN) return;

	// CudaDbgPrn ("after comparing strlen idx %u, row %u, str %s", (unsigned) idx, (unsigned int) mp -> d_table [idx].row, mp -> d_table [idx].element);
	// strlen is the same here so...

#pragma unroll
	for (i = 0; i < CONST_TLEN; i++) {
		if (mp -> d_table [idx].element [i] != d_patt [i]) {
			mp -> result [gbl_th] = (uint64_cu) 0;
			return;
		}
	}

	// CudaDbgPrn ("before the true tab[%lu] th %u", (uint32_cu) mp -> d_table [idx].row, (unsigned) gbl_th);

	mp -> result [gbl_th] = (uint64_cu) mp -> d_table [idx].row;

	// CudaDbgPrn ("gbl_th %u, thIdx.x %u, %u: %s", (unsigned) gbl_th, (unsigned) threadIdx.x, (unsigned int) mp -> d_table [idx].row, mp -> d_table [idx].element);

	atomicAdd ((uint64_cu *) roc, (uint64_cu) 1);	// result_on_card
	__threadfence ();
	// CudaDbgPrn ("true row %u result %u, gbl_th %u roc %u", (unsigned) mp -> d_table [idx].row, (unsigned) mp -> result [gbl_th], (unsigned) gbl_th, (unsigned) *roc);
}

static __global__ void K_compress_result (mi_params_for_kernel1 *mp, uint64_cu *d_c_res)
{
	// constant declarations at the top of the file
	// #define CONST_MROW	d_mrow [0] max_rows
	// mt, fid, tlen fcnt, d_mrow;

	const uint64_cu gbl_th = (uint64_cu) (blockIdx.x * blockDim.x + threadIdx.x);
	if (gbl_th >= CONST_MROW) return;

	// register bool leaveLoop = false;
	unsigned long long int l_id;
	uint64_cu res = (uint64_cu) mp -> result [gbl_th];
	if (0 == res) return;

	l_id  = (unsigned long long int) atomicAdd ((unsigned long long int *) id, (unsigned long long int) 1);

	d_c_res [(unsigned long long int) l_id] = (uint64_cu) res;


	// CudaDbgPrn ("mp -> result [gbl_th %u] = %u", (unsigned) gbl_th, (unsigned) mp -> result [(unsigned) gbl_th]);

	// CudaDbgPrn ("res %u, id %lu gbl_th %u, blockIdx.x %u, threadIdx.x %u", (unsigned) res, *id, (unsigned) gbl_th, (unsigned) blockIdx.x,  (unsigned) threadIdx.x);
	// CudaDbgPrn ("d_c_res %lu, l_id %u", (unsigned long) d_c_res [(unsigned long long int) l_id], (unsigned) l_id);
}

/*******************************************************************
void test_table (uint64_cu r_count, mi_sqlres *res, gpu_config *cfg)
{
	dim3 block (cfg -> dev_prop -> maxThreadsPerBlock / 2);
	dim3 grid ((r_count + block.x - 1)/block.x);

	Dbg ("before kernel_test_table r_count %llu", r_count);
	kernel_test_table <<<grid, block>>> (res -> d_table);
	gpuErrChk (cudaPeekAtLastError ());
	gpuErrChk (cudaDeviceSynchronize());
}

static __global__ void K_test_table (mi_sqltable *tab)
{
	uint64_cu gbl_th = (blockIdx.x * blockDim.x + threadIdx.x) % CONST_MT;
	register uint64_cu	idx = CONST_FCNT * gbl_th + CONST_FID;
	__syncthreads ();

#pragma unroll 1
	for (unsigned int i = 0; i < CONST_FCNT; i++) {
		if (!(gbl_th % 10000)) {
			CudaDbgPrn ("len: %u, fld: %u, row: %u elem: %s", (unsigned) tab [idx].length, (unsigned) tab [idx].fld, (unsigned) tab [idx].row, tab [idx].element);
		}
	}
}
*******************************************************************/