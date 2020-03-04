	// %{
	// C DECLARATIONS
	// %}
%{

#include  "sql_query.hpp"

extern  int  yylex ();
extern  void  yyerror (connection_details *sqld, const char *);

extern char *yytext;
bool		error_situation = false;

extern mi_sqltable *create_cuda_buffer (mi_sqlres *res);
extern void free_all_memory (mi_sqlres *tab, connection_details *mysqlD);
extern void GetResultofSQLQueryfromGPU (MYSQL *conn, struct connection_details *mysqlD);
extern void GetResultofSQLQueryfromDB (MYSQL *conn, struct connection_details *mysqlD);

%}

	// BISON DECLARATIONS
%union {
	int      int_token;
}

%parse-param {connection_details *sqld}

	// list  the  different  tokens  of each  type
%token <int_token>	TOK_EXIT  TOK_STRING  TOK_HELP  TOK_SET  TOK_ALL  TOK_CACHE  TOK_GPU  TOK_TIME  TOK_ON  TOK_OFF  TOK_SELECT  TOK_STAR  TOK_FROM  TOK_EQUALS  TOK_WHERE

%start Ready_Node


	// %%
	// GRAMMAR RULES
	// %%
%%
Ready_Node:			Set_Node	Ready_Node;
				|	Cache_On_Node	Ready_Node;
				|	Cache_Off_Node	Ready_Node;
				|	Set_GPU_On_Node	Ready_Node;
				|	Set_GPU_Off_Node	Ready_Node;
				|	Set_Time_On_Node	Ready_Node;
				|	Set_Time_Off_Node	Ready_Node;
				|	Select_Node Ready_Node;
				|	Where_Node	Ready_Node;
				|	Equals_Node	Ready_Node;
				|	String_Node	Ready_Node;
				|	Help_Node	Ready_Node;
				|	Exit_Node;

Set_Node:			TOK_SET TOK_ALL
	{
		(sqld -> cache) ? printf ("cacheing enabled\n") : printf ("cacheing disabled\n");
		(sqld -> gpu) ? printf ("gpu enabled\n") : printf ("cpu enabled\n");
		(sqld -> time) ? printf ("time measurement enabled") : printf ("time measurement disabled");
		printf ("\nMI_Data > ");
	};

Cache_On_Node:		TOK_SET TOK_CACHE TOK_ON

	{
		printf ("Enabling Cache for CPU based queries");
		sqld -> cache = true;
		printf ("\nMI_Data > ");
	};

Cache_Off_Node:		TOK_SET TOK_CACHE TOK_OFF
	{
		printf ("Disabling Cache for CPU based queries");
		sqld -> cache = false;
		printf ("\nMI_Data > ");
	};

Set_GPU_On_Node:	TOK_SET TOK_GPU TOK_ON
	{
		printf ("Enabling GPU processing");
		sqld -> gpu = true;
		printf ("\nMI_Data > ");
	};

Set_GPU_Off_Node:	TOK_SET TOK_GPU TOK_OFF
	{
		printf ("Enabling CPU based processing");
		sqld -> gpu = false;
		printf ("\nMI_Data > ");
	};

Set_Time_On_Node:	TOK_SET TOK_TIME TOK_ON
	{
		printf ("Enabling time measurement for queries");
		sqld -> time = true;
		printf ("\nMI_Data > ");
	};

Set_Time_Off_Node:	TOK_SET TOK_TIME TOK_OFF
	{
		printf ("Disabling time measurement for queries");
		sqld -> time = false;
		printf ("\nMI_Data > ");
	};

Select_Node:		TOK_SELECT TOK_STAR TOK_FROM TOK_STRING
	{
		// if the table is wrong then error.
		if (0 != strcmp (sqld -> table, yytext)) {	// new table
			printf ("Invalid table\nExit and restart MiSQL to load new table\n");
			printf ("Alternatively try again\n");
			error_situation = true;
			printf ("\nMI_Data > ");
		}

		/**********
		if (false == error_situation) {
			Dbg ("Table: %s", sqld -> table);
		}
		**********/
	};

Where_Node:			TOK_WHERE TOK_STRING
	{
		uint32_cu i = 0;
		if (false == error_situation) {
			// Get the field name
			strcpy ((char *) sqld -> sql -> field, yytext);
			sqld -> sql -> fld_len = strlen (yytext);
			sqld -> sql -> field [sqld -> sql -> fld_len] = '\0';

			for (i = 0; i < sqld -> res -> field_count; i++) {
				if (0 == strcmp ((char *) (sqld -> res -> field + i) -> name, (char *) sqld -> sql -> field)) {
					sqld -> res -> field_id = i;	// we got the field_id
					break;
				}
			}

			if (i >= sqld -> res -> field_count) {
				Dbg ("Field not found");
				error_situation = true;
			}
		}
	};

Equals_Node:		TOK_EQUALS TOK_STRING
	{
		if (false == error_situation) {
			char number_of_records [TABLE_NAME];
			if (0 != sqld -> res -> row_count) {
				sprintf (number_of_records, "limit %llu", (long long unsigned) sqld -> res -> row_count);
			}
			else {
				number_of_records [0] = 0x20;	// space
				number_of_records [1] = '\0';
			}

			// get the string and start execution
			strcpy ((char *) sqld -> sql -> str_tobe_matched, yytext);
			sqld -> sql -> str_len = strlen (yytext);
			sqld -> sql -> str_tobe_matched [sqld -> sql -> str_len] = '\0';

			if (false == sqld -> gpu) {
				if (false == sqld -> cache) {
					sprintf ((char *) sqld -> sql -> full_query, "select SQL_NO_CACHE * from (select * from %s %s) t where %s = %s", sqld -> table, number_of_records, sqld -> sql -> field, sqld -> sql -> str_tobe_matched);
				}
				else {
					sprintf ((char *) sqld -> sql -> full_query, "select * from (select * from %s %s) t where %s = %s", sqld -> table, number_of_records, sqld -> sql -> field, sqld -> sql -> str_tobe_matched);
				}

				GetResultofSQLQueryfromDB (sqld -> conn, sqld);	// cpu
			}
			else {	// on GPU
				// use the already chunked query to trigger the gpu.
				GetResultofSQLQueryfromGPU (sqld -> conn, sqld);	// gpu
			}
		}

		printf ("\nMI_Data > ");
	};

String_Node:		TOK_STRING
	{
		printf ("Received STRING %s", yytext);
		printf ("\nMI_Data > ");
	};

Help_Node:			TOK_HELP
	{
		system ("cat ./help_file.txt");
		printf ("\nMI_Data > ");
	};

Exit_Node:			TOK_EXIT
	{
		free_all_memory (sqld -> res, sqld);
		exit (0);
	};

%%

	// ADDITIONAL C CODE
