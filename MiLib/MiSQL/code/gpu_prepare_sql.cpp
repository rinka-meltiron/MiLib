#include "../include/cuda_milib.h"
#include "sql_query.hpp"

static uint32_cu create_field (connection_details *res_tab, MYSQL_RES *mRes);
static void load_table_into_gpu (connection_details *cd, MYSQL_RES *res);

extern __global__ void kernel_test_table (mi_sqltable *tab, unsigned char *data);

extern "C" mi_sqlres *setup_and_extract_database (MYSQL *conn, struct connection_details *mysqlD);
extern "C" void free_all_memory (mi_sqlres *tab, connection_details *mysqlD);
extern "C" uint32_cu add_and_cp_to_buffer (mi_sqlres *mi_res, MYSQL_ROW row, mi_sqltable *hd_local_row, uint64_cu num_row);
extern "C" mi_sqltable *create_cuda_buffer (mi_sqlres *res);

mi_sqltable *sql_fetch_result (mi_sqlres *res, const uint64_cu res_rows, bool new_query);

/*******************************************************************/
static void load_table_into_gpu (connection_details *cd, MYSQL_RES *res)
{
	uint64_cu	row_num;	// row number
	MYSQL_ROW	row;		// results row (line by line)

	// this is getting the whole table
	mi_sqltable *hd_local_row = create_cuda_buffer (cd -> res);
	if (!hd_local_row) return;

	for (row_num = 1; (NULL != (row = mysql_fetch_row (res))) && (row_num <= cd -> res -> row_count); row_num++) {
		// create new row on cpu & GPU
		add_and_cp_to_buffer (cd -> res, row, hd_local_row, row_num);
	}
	printf ("\n");

	FREE (hd_local_row);
}

/*******************************************************************/
static uint32_cu create_field (connection_details *res_tab, MYSQL_RES *mRes)
{
	uint32_cu i;
	MYSQL_FIELD *fld;

	for(i = 0; i < res_tab -> res -> field_count; i++) {
		fld = mysql_fetch_field_direct (mRes, i);
		(res_tab -> res -> field + i) -> length = fld -> length;

		(res_tab -> res -> field + i) -> name = (unsigned char *) calloc (sizeof (unsigned char) * fld -> length, sizeof (unsigned char));
		check_mem ((res_tab -> res -> field + i) -> name);

		strncpy ((char *) (res_tab -> res -> field + i) -> name, fld -> name, fld -> length);

		// Dbg ("%s: %u", (res_tab -> field + i) -> name, (unsigned int) (res_tab -> field + i) -> length);
	}

	return SUCCESS;

error:
	return FAILED;
}

/*******************************************************************/
mi_sqlres *setup_and_extract_database (MYSQL *conn, connection_details *mysqlD)
{
	char number_of_records [TABLE_NAME];
	if (mysqlD -> res -> row_count) {
		sprintf (number_of_records, "limit %llu", (long long unsigned) mysqlD -> res -> row_count);
	}
	else {
		number_of_records [0] = 0x20;	// space
		number_of_records [1] = '\0';
	}

	char		select_part [SQL_QUERY];
	sprintf (select_part, "select * from %s %s", mysqlD -> table,  number_of_records);
	if (mysql_query (conn, select_part)) { // get db
		Dbg ("MySQL query error: %s", mysql_error(conn));
		return FAILED;
	}

	MYSQL_RES	*mRes = mysql_store_result (conn);

	mysqlD -> res -> row_count = mysql_num_rows (mRes);
	if (mysqlD -> sql -> row_limit < mysqlD -> res -> row_count) {
		mysqlD -> res -> row_count = mysqlD -> sql -> row_limit;
	}

	mysqlD -> res -> field_count = mysql_field_count (conn);

	mysqlD -> res -> field = (mi_sqlfield *) calloc (mysqlD -> res -> field_count * sizeof (mi_sqlfield), sizeof (mi_sqlfield));
	check_mem (mysqlD -> res -> field);

	mysqlD -> res -> db_bytes = get_full_db_size (conn, mysqlD, mysqlD -> sql -> row_limit);
	mysqlD -> res -> db_bytes += mysqlD -> res -> db_bytes * THIRTY_PERCENT; // for buffer
	// OK. We have gotten the total table size & stored it.
	// Dbg ("DB size %u %uMB", (unsigned) mysqlD -> res -> db_bytes, (unsigned) mysqlD -> res -> db_bytes/1000000);

	if (FAILED == create_field (mysqlD, mRes)) goto error; // get field names

	// load the table into gpu & memory
	printf ("Loading table\n");
	load_table_into_gpu (mysqlD, mRes);
	mysql_free_result (mRes);

	return mysqlD -> res;

error:
	mysql_free_result (mRes);
	return FAILED;
}

// Non reentrant and can be called only by one thread.
mi_sqltable *sql_fetch_result (mi_sqlres *res, const uint64_cu res_rows, bool new_query)
{
	static uint64_cu	j = 0;	// total rows

	static mi_sqltable	*row = NULL;
	static uint64_cu 	row_id = 0;
	static uint64_cu	prev_row_id = 0xffffffff;
	if (new_query) {
		j = 0;
		row_id = 0;
		prev_row_id = 0xffffffff;
	}

	if (j >= res_rows) {
		FREE (row);
		j++;
		goto error;
	}

	// Dbg ("Found Rows %u", (unsigned) res_rows);
	row_id = (uint64_cu) res -> h_c_res [j];
	// Dbg ("row_id %llu", res -> h_c_res [j] + 1);

	if (prev_row_id != row_id) {
		if (!row) {
			row = (mi_sqltable *) calloc (res -> field_count * sizeof (mi_sqltable), sizeof (mi_sqltable));
			check_mem (row);
		}

		for (uint32_cu i = 0; i < res -> field_count; i++) {
			row [i].element = (res -> h_table + ((row_id - 1) * res -> field_count + i)) -> element;
			row [i].fld = (res -> h_table + ((row_id - 1) * res -> field_count + i)) -> fld;
			row [i].row = row_id;
			row [i].length = (res -> h_table + ((row_id - 1) * res -> field_count + i)) -> length;

			// Dbg ("curr: str %s row %u", row [i].element, (unsigned) row [i].row);
		}

		prev_row_id = row_id;

		j++;
		return row;
	}

error:
	j++;
	FREE (row);
	return NULL;
}

void free_all_memory (mi_sqlres *tab, connection_details *mysqlD)
{
	printf ("\nExiting...\n");
	mysql_close(mysqlD -> conn);	// clean database link
	int i = CUDA_SUCCESS;
	if (CUDA_SUCCESS != (i = cuCtxDestroy (mysqlD -> cfg -> cuc))) {
		Dbg ("cuCtxDestroy failed with %u", (unsigned) i);
	}
	gpuErrChk (cudaDeviceReset ());

	FREE (mysqlD -> sql -> field);
	FREE (mysqlD -> sql -> str_tobe_matched);
	FREE (mysqlD -> sql -> full_query);
	FREE (mysqlD -> sql);

	FREE (mysqlD -> cfg -> dev_prop);
	FREE (mysqlD -> cfg);

	FREE (mysqlD);

	// tab mi_sqlres
	if (tab) {
		for (uint32_cu i = 0; i < tab -> field_count; i++) {
			FREE ((tab -> field + i) -> name);
		}
		FREE (tab -> field);

		FREE (tab -> h_table);
		FREE (tab -> h_data);
		FREE (tab);
	}
}
