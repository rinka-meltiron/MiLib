	// user code
%option case-insensitive nodefault

	// definitions
%{
#include  "sql_query.hpp"
#include "cmds.tab.h"	// the  output  of  bison on  cmds.y

	// forward  declaring  functions
void  yyerror (char *);
int  yyparse (connection_details *sqld);

extern bool	error_situation;
%}

	// %%
	// rules
	// %%
%%
"SET"			{ error_situation = false; return TOK_SET; };
"ALL"			{ return TOK_ALL; };
"CACHE"			{ return TOK_CACHE; };
"GPU"			{ return TOK_GPU; };
"TIME"			{ return TOK_TIME; };
"ON"			{ return TOK_ON; };
"OFF"			{ return TOK_OFF; };
"SELECT"		{ error_situation = false; return TOK_SELECT; };
"="				{ return TOK_EQUALS; };
"*"				{ return TOK_STAR; };
"\""			;
"FROM"			{ return TOK_FROM; };
"WHERE"			{ return TOK_WHERE; };
"HELP"			{ error_situation = false; return TOK_HELP; };
"EXIT"			{ return TOK_EXIT; };

[ \t\n]			;
[a-zA-Z0-9_]+	{ return TOK_STRING; };
.				{ printf("bad input character '%s' at line %d\n", yytext, yylineno); printf ("\nMI_Data > "); };
%%

void  yyerror (char* str) {printf("ERROR: Could  not  parse %s !\n", str);};
int  yywrap (void) {};
