#include <mysql.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <ctype.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <signal.h>

#include "../include/milib.h"

#ifndef _SQL_QUERY_HPP_
#define _SQL_QUERY_HPP_

#define	SERVER_SIZE		199
#define	ROW_NUM_LIMIT	10000000
// #define	ROW_NUM_LIMIT	30000000
#define	USER_NAME		80
#define	PASSWORD		80
#define	DB_NAME			255
#define	SQL_QUERY		512
#define	TABLE_NAME		80

#define CONST_PARAMS0	0
#define CONST_PARAMS1	1
#define CONST_PARAMS2	2
#define CONST_PARAMS3	3
#define CONST_PARAMS4	4
#define CONST_PARAMS5	5
#define CONST_PARAMS6	6
#define CONST_PARAMS7	7
#define CONST_PARAMS8	8

#define LOCATION_OF_DATABASE_SIZE		3
#define LOCATION_OF_ROW_SIZE			2

#define 	QTABLE_SIZE		"SELECT table_schema as `Database`, table_name AS `Table`, table_rows AS `Rows`, round((data_length + index_length) , 2) `Size in Bytes` FROM information_schema. TABLES ORDER BY (data_length + index_length) DESC;"

// #define	STR_TO_SRCH		"38849"
// #define	STR_TO_SRCH		"60117"
// #define	STR_TO_SRCH		"Vadodara"

typedef enum {field, string}	sql_chunk;

typedef struct mi_sql {
	uint64_cu	row_limit;
	unsigned char		*full_query;

	// parsed data
	unsigned char	*field;
	unsigned int	fld_len;
	unsigned char	*str_tobe_matched;
	unsigned int	str_len;
} mi_sql;

typedef struct connection_details {
	char	server [SERVER_SIZE];
	char	database [DB_NAME];
	char	table [DB_NAME];
	char	field_name [DB_NAME];
	unsigned int	tab_len;

	mi_sql	*sql;
	struct mi_sqlres		*res;
	struct gpu_config		*cfg;
	struct mi_params_for_kernel1	*d_p;
	MYSQL					*conn;

	bool	cache;			// global values
	bool	gpu;
	bool	time;
} connection_details;

typedef struct mi_sqltable {
	uint32_cu	length;
	uint32_cu	fld;
	uint64_cu	row;
	unsigned char		*element;
} mi_sqltable;

// typedef struct __attribute__ ((packed)) mi_sqlres {
// typedef struct ALIGN(8) mi_sqlres {
typedef struct mi_sqlres {
	mi_sqltable		*h_table;
	mi_sqltable		*d_table;		// table on gpu
	struct mi_sqlfield	*field;
	unsigned char	*d_data;		// data ptd to by table
	unsigned char	*h_data;

	uint32_cu	field_count;
	uint32_cu	field_id;
	uint64_cu	db_bytes;
	uint64_cu	row_count;

	uint64_cu	*d_res;	// result of srch
	uint64_cu	*d_c_res;	// compressed result
	uint64_cu	*h_c_res;
} mi_sqlres;

typedef struct mi_sqlfield {
	unsigned char	*name;	// Name of column
	uint32_cu	length;	// Width of column
} mi_sqlfield;

typedef struct mi_params_for_kernel1 {
	mi_sqltable			*d_table;
	uint64_cu	*result;
} mi_params_for_kernel1;

/********************************
 * code from: http://www.codingfriends.com/index.php/2010/02/17/mysql-connection-example/
 * Data from: https://archive.org/details/stackexchange
 * mi_* structures from mysql.h
 *
 * Look at: http://docstore.mik.ua/orelly/linux/sql/ch19_01.htm for mysql api descriptions
 * time: http://stackoverflow.com/questions/1468596/calculating-elapsed-time-in-a-c-program-in-milliseconds
 ********************************/

// Definitions from lib_func.cu
extern int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
extern uint64_cu get_full_db_size (MYSQL *conn, struct connection_details *mysqlD,  const uint64_cu limit_rows);
extern uint64_cu asc_to_uint64_cu (const char *txt);
extern inline unsigned maxThreads(uint64_cu th, const unsigned unroll);

#ifdef _CUDA_MILIB_H_
// Definitions from lib_func.cu
__device__ __host__ inline uint32_cu int_div_rounded (uint64_cu num, const unsigned div);
extern __device__ void cuda_strn_cpy (unsigned char *tgt, const unsigned char *src, const unsigned len);
extern __device__ unsigned cuda_string_len (char *str);
extern __device__ void acquire_semaphore (volatile int *lock);
extern __device__ void release_semaphore (volatile int *lock);
#endif // _CUDA_MILIB_H_

#endif // _SQL_QUERY_HPP_
