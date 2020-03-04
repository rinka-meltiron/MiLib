/* MiSQL - Melt Iron SQL acceleration - Prototype
 * Copyright (C) 2016 Rinka Singh
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../include/cuda_milib.h"
#include "sql_query.hpp"

uint64_cu get_full_db_size (MYSQL *conn, struct connection_details *mysqlD,  const uint64_cu limit_rows);
static uint64_cu parse_db_bytes (char **row, char *db, uint32_cu db_sz_loc);
uint64_cu asc_to_uint64_cu (const char *txt);
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
inline unsigned maxThreads(uint64_cu th, const unsigned unroll);

__device__ __host__ inline uint32_cu int_div_rounded (uint64_cu num, const unsigned div);

__device__ void cuda_strn_cpy (unsigned char *tgt, const unsigned char *src, const unsigned len);
__device__ unsigned cuda_string_len (char *str);

__device__ void acquire_semaphore (volatile int *lock);
__device__ void release_semaphore (volatile int *lock);

#define LOCATION_OF_DATABASE_SIZE		3
#define LOCATION_OF_ROW_SIZE			2

#define 	QTABLE_SIZE		"SELECT table_schema as `Database`, table_name AS `Table`, table_rows AS `Rows`, round((data_length + index_length) , 2) `Size in Bytes` FROM information_schema. TABLES ORDER BY (data_length + index_length) DESC;"
uint64_cu get_full_db_size (MYSQL *conn, struct connection_details *mysqlD,  const uint64_cu limit_rows)
{
	MYSQL_RES	*res;
	MYSQL_ROW	row; // the results row
	uint64_cu	num_of_rows;

	if (mysql_query (conn, QTABLE_SIZE)) {
		Dbg ("MySQL query error: %s", mysql_error(conn));
		return FAILED;
	}

	res = mysql_store_result (conn);

	while((row = mysql_fetch_row (res)) != NULL) {
		int64_cu db_bytes = parse_db_bytes (row, mysqlD -> database, LOCATION_OF_DATABASE_SIZE); // Table sz

		// Dbg ("DB size %u", (unsigned) db_bytes);
		if (db_bytes) {
			num_of_rows = asc_to_uint64_cu (row [LOCATION_OF_ROW_SIZE]);
			// Dbg ("Num rows %u", (unsigned) num_of_rows);

			if (limit_rows < num_of_rows) {
				db_bytes *= ((double) limit_rows / num_of_rows);
			}

			mysql_free_result (res);
			return db_bytes;
		}
	}

	return FAILED;
}

static uint64_cu parse_db_bytes (char **row, char *db, uint32_cu db_sz_loc)
{
	if (0 == strcmp (row [0], db)) {
		// Dbg ("%s %s %s", row [0], row [1], row [db_sz_loc]);
		return asc_to_uint64_cu (row [db_sz_loc]);
	}

	return FAILED;
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

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}


// C/C++
inline unsigned maxThreads(uint64_cu th, const unsigned unroll)
{
	return ((th % unroll) ? (th / unroll + 1): (th / unroll));
}


__device__ __host__ inline uint32_cu int_div_rounded (uint64_cu num, const unsigned div)
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
	for (i = 0; i < len; i++) tgt [i] = src [i];
}

__device__ unsigned cuda_string_len (char *str)
{
	unsigned	i = 0;
	char *s = str;
	for (; s && *s; ++s) ++i;

	return (i);
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
	__syncthreads ();
}
