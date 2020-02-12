#include <termios.h>
#include "cuda_milib.h"
#include "sql_query.hpp"
#include "cmds.tab.h"	// the  output  of  bison on  cmds.y

extern gpu_config *setup_gpu_card (void);
extern mi_sqlres *setup_and_extract_database (MYSQL *conn, struct connection_details *mysqlD);
extern void load_params_into_card1 (mi_sqlres *tab, uint64_cu tth, mi_sql *query);
extern int GetResultofSQLQueryfromDB (MYSQL *conn, struct connection_details *mysqlD);
extern void GetResultofSQLQueryfromGPU (MYSQL *conn, struct connection_details *mysqlD);
extern void free_all_memory (mi_sqlres *tab, connection_details *mysqlD);

extern int yyparse (connection_details *sqld);

extern gpu_config *setup_gpu_card (void);

static MYSQL* mysql_connection_setup (connection_details *mysqlD);
static connection_details *create_mi_sql_buffers (void);

/*******************************************************************/
static MYSQL* mysql_connection_setup (connection_details *mysqlD)
{
	char	user [USER_NAME];
	char	password [PASSWORD];

	MYSQL *connection = mysql_init(NULL);	// create mysql instance and init variables
	check_mem (connection);

	// setting default params for the connection
	mysqlD -> cache = false;
	mysqlD -> gpu = false;
	mysqlD -> time = true;

	int i;
	for (i = 0; i < 3; i++) {
		printf ("Server address: localhost");
		// scanf ("%s", mysqlD -> server);		// where the mysql database is "localhost"
		strcpy (mysqlD -> server, "localhost");

		printf ("\nLogin id: root");
		// scanf ("%s", user);					// the root user of mysql
		strcpy (user, "root");


		strcpy (password, "rinka123");	// password of root
		printf ("\nPassword: ");
		// strcpy (password, getpass (" "));	// the password

		// DEBUG - emp vs Vadodara
		// strcpy (mysql_details -> database, "UserInfo");	// the databse to pick
		printf ("\nDatabase: employees");
		// scanf ("%s", mysqlD -> database);	// the database
		strcpy (mysqlD -> database, "employees");	// the database to pick

		printf ("\nTable: salaries");
		strcpy (mysqlD -> table, "salaries");	// the table
		// scanf ("%s", mysqlD -> table);	// the table
		mysqlD -> tab_len = strlen (mysqlD -> table);
		mysqlD -> table [mysqlD -> tab_len] = '\0';

		mysqlD -> sql -> row_limit = ROW_NUM_LIMIT;
		printf ("\nRows of DB to read (0 for all): 0\n");
		// scanf ("%ul", mysqlD -> sql -> row_limit);	// the number of rows

		// connect to the database with the details attached.
		if (mysql_real_connect (connection, mysqlD -> server, user, password, mysqlD -> database, 0, NULL, 0)) {
			printf ("Db Connected\n");
			return connection;
		}
		else {
			Dbg ("Couldn't connect DB, please retry...");
		}
	}

	Dbg ("Db Connection failed : %s", mysql_error (connection));
error:
	return FAILED;
}

/*******************************************************************/
static connection_details *create_mi_sql_buffers (void)
{
	connection_details *mysqlD = (connection_details *) calloc (sizeof (connection_details), sizeof (connection_details));
	check_mem (mysqlD);

	mysqlD -> sql = (mi_sql *) calloc (sizeof (mi_sql), sizeof (mi_sql));
	check_mem (mysqlD -> sql);

	mysqlD -> sql -> full_query = (unsigned char *) calloc (sizeof (unsigned char) * SQL_QUERY, sizeof (unsigned char));
	check_mem (mysqlD -> sql -> full_query);

	mysqlD -> sql -> field = (unsigned char *) calloc (sizeof (unsigned char) * TABLE_NAME, sizeof (unsigned char));
	check_mem (mysqlD -> sql -> field);

	mysqlD -> sql -> str_tobe_matched = (unsigned char *) calloc (sizeof (unsigned char) * TABLE_NAME, sizeof (unsigned char));
	check_mem (mysqlD -> sql -> str_tobe_matched);

	mysqlD -> res = (mi_sqlres *) malloc (sizeof (mi_sqlres));
	check_mem (mysqlD -> res);

	return mysqlD;

error:
	FREE (mysqlD -> sql -> field);
	FREE (mysqlD -> sql -> str_tobe_matched);
	FREE (mysqlD -> sql);
	FREE (mysqlD);

	return FAILED;
}

/*******************************************************************/
int main (void)
{
	connection_details *mysqlD = create_mi_sql_buffers ();
	if (FAILED == mysqlD) goto error;

	mysqlD -> cfg = setup_gpu_card ();				// gpu config
	if (FAILED == mysqlD -> cfg) goto error;

	mysqlD -> conn = mysql_connection_setup (mysqlD);	// connect to db
	if (FAILED == mysqlD -> conn) goto error;

	mysqlD -> res = setup_and_extract_database (mysqlD -> conn, mysqlD);	// gpu
	if (FAILED == mysqlD -> res) goto error;

	printf ("MI_Data > ");
	yyparse (mysqlD);				//  parser: defined by flex

	return 0;

error:
	free_all_memory (mysqlD -> res, mysqlD);

	return 1;
}