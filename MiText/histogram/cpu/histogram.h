#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>

#define WORD_SIZE		256

typedef enum {INSERT_BEFORE_HEAD, INSERT_AFTER_TAIL, FOUND, NOT_FOUND}	where;
typedef enum {FORWARD, BACK, HERE}	direction;

typedef struct token {
	uint64_t		num;
	uint64_t		hash;

	char			*tk_str;
	struct token	*next_token;
	struct token	*prev_token;
	struct token	**phead;
} token;

typedef struct math_results {
	double	std_dev;

	token	*avg;
	token	*mean;
	token	*min;
	token	*max;
} math_results;

// global variable - to wrap locks around it
token	*list_head;