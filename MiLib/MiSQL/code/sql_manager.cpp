#include <sys/time.h>
#include "sql_query.hpp"

uint64_cu get_full_db_size (MYSQL *conn, struct connection_details *mysqlD,  const uint64_cu limit_rows);
static uint64_cu parse_db_bytes (char **row, char *db, uint32_cu db_sz_loc);
extern "C" int GetResultofSQLQueryfromDB (MYSQL *conn, struct connection_details *mysqlD);

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

// char * tableName, char *columnName, char* operator, char* columnValue
int GetResultofSQLQueryfromDB (MYSQL *conn, struct connection_details *mysqlD)
{
	MYSQL_RES *res;		// the results
	MYSQL_ROW row;		// the results row (line by line)
	MYSQL_FIELD	*fld;	// the fields

	struct timeval	t1, t2, tvDiff;
	uint32_cu	i;

	gettimeofday (&t1, NULL);
	if (mysql_query(conn, (char *) mysqlD -> sql -> full_query)) {	// send query to the database
		Dbg ("MySQL query error : %s\n", mysql_error(conn));
		return FAILED;
	}
	gettimeofday (&t2, NULL);

	res = mysql_use_result(conn);

	while((row = mysql_fetch_row(res)) != NULL) {
		for(i = 0; i < mysqlD -> res -> field_count; i++) {
			fld = mysql_fetch_field_direct (res, i);
			printf ("%s: %s, ", fld -> name, row[i]);
		}
		printf ("\n");
	}

	if (mysqlD -> time) {
		timeval_subtract(&tvDiff, &t2, &t1);
		Dbg ("Time taken by query %ld.%06ld sec\n", tvDiff.tv_sec, tvDiff.tv_usec);
	}

	mysql_free_result(res);
	return SUCCESS;
}
