#include <sys/types.h>
#include <stdlib.h>

#include "histogram.h"
#include "../../include/milib.h"

char *next_token (FILE *fl);
token *create_list ();
direction histo_strcmp (const char *str1, const char *str2);
where found_loc_in_list (token **list, char *tok);
void increment_token (token *list);
void update_token (token *list, char *str);
uint64_t hash_func (char *str);
void insert_token (where pos, token *loc, char *str);
void insert_token_into_list (token **list, char *tok);
math_results *math_process_list (token *list);
void print_histogram (math_results *res, token *lst);


char *next_token (FILE *fl)
{
	char wd [WORD_SIZE];
	int ch;
	size_t idx;

	for (idx = 0; idx < WORD_SIZE; idx++) {
		ch = fgetc (fl);
		if (ch == EOF) return NULL;

		if (!isalpha (ch)) {
			if (!idx) continue;				// Nothing read yet, skip character
			else break;						// gone beyond the current word
		}

		wd [idx] = tolower (ch);
	}

	if (!idx) return NULL;					// no chars left
	wd [idx] = '\0';
	return strdup (wd);
}

token *create_list ()
{
	token *lst = malloc (sizeof (token));
	check_mem (lst);

	lst -> num = lst -> hash = 0;
	lst -> next_token = lst -> prev_token = NULL;
	lst -> phead = &list_head;
	list_head = lst;

	return lst;

error:
	exit (1);
}

// compare str1 & str2 if str1>str2 ret: FORWARD, str1<str2 ret: BACK
// if match ret: FOUND, if str1>str2 & first chars match, ret: FORWARD
// and so on.
direction histo_strcmp (const char *str1, const char *str2)
{
	int lstr1 = strlen (str1);
	int lstr2 = strlen (str2);

	for (;*str1 && *str2; str1++, str2++) {
		if (*str1 < *str2) return BACK;
		if (*str1 > *str2) return FORWARD;
	}

	if (lstr1 > lstr2) return FORWARD;
	if (lstr1 < lstr2) return BACK;

	return HERE;
}

where found_loc_in_list (token **list, char *tok)
{
	token *curr = *list;					// current loc
	static bool fwd = false, back = false;	// locating where to insert

	if (!tok || !curr -> tk_str) {
		Dbg ("str1 or str2 does not exist");
		exit (1);
	}

	while (true) {
		direction mv = histo_strcmp (tok, curr -> tk_str);

		if (HERE == mv) {
			*list = curr;
			fwd = back = false;
			return FOUND;
		}
		else if (BACK == mv) {			// go back
			back = true;
			if (true == fwd) {			// insert after current
				*list = curr -> prev_token;
				fwd = back = false;
				return NOT_FOUND;		// insert here
			}

			if (NULL == curr -> prev_token) {	// not there at head
				*list = curr;
				fwd = back = false;
				return INSERT_BEFORE_HEAD;	// insert here
			}
			curr = curr -> prev_token;	// go back & loop again
		}
		else if (FORWARD == mv) {
			fwd = true;
			if (true == back) {			// insert after current
				*list = curr;
				fwd = back = false;
				return NOT_FOUND;		// insert here
			}

			if (NULL == curr -> next_token) {
				*list = curr;
				fwd = back = false;
				return INSERT_AFTER_TAIL;
			}
			curr = curr -> next_token;
		}
	}
}

void increment_token (token *list)
{
	++list -> num;
	Dbg ("%i", (int) list -> num);
}

void update_token (token *list, char *str)
{
	increment_token (list);
	list -> tk_str = str;
	Dbg ("%s", list -> tk_str);
}

uint64_t hash_func (char *str)
{
	uint64_t hash = 0;
	for (;*str; str++) hash += (uint64_t) *str;

	return hash;
}

void insert_token (where pos, token *loc, char *str)
{
	token *tmp = malloc (sizeof (token));
	check_mem (tmp);

	tmp -> tk_str = str;
	tmp -> num = 0;
	tmp -> hash = hash_func (str);
	tmp -> phead = &list_head;
	increment_token (tmp);

	switch (pos) {
		case INSERT_BEFORE_HEAD:
			tmp -> next_token = loc;
			tmp -> prev_token = NULL;
			loc -> prev_token = tmp;
			list_head = tmp;			// this is now the head
			Dbg ("inserted %s before head", str);
			break;

		case INSERT_AFTER_TAIL:
			loc -> next_token = tmp;
			tmp -> prev_token = loc;
			tmp -> next_token = NULL;
			Dbg ("inserted %s after tail", str);
			break;

		case NOT_FOUND:
			tmp -> next_token = loc -> next_token;
			tmp -> prev_token = loc;
			loc -> next_token -> prev_token = tmp;
			loc -> next_token = tmp;
			Dbg ("inserted new token %s after %s", str, loc -> tk_str);
			break;

		case FOUND:			// major error
			Dbg ("FOUND should not happen here");
			exit (1);
	}
	return;

error:
	exit (1);
}

void insert_token_into_list (token **list, char *tok)
{
	where	loc = found_loc_in_list (list, tok);	// srch for token

	if (FOUND == loc) {						// list has current loc.
		increment_token (*list);
		FREE (tok);							// reduce the mallocs.  Done because I use strdup.
	}
	else {									// Not found in list
		insert_token (loc, *list, tok);		// new token hence create & insert
	}
}

math_results *math_process_list (token *list)
{
	goto error;

error:
	return NULL;
}

void print_histogram (math_results *res, token *lst)
{

}

int main ()
{
	FILE			*in_file;
	char			*tok = NULL;
	math_results	*res = NULL;
	token			*list = NULL;

	in_file = fopen ("data.txt", "r");
	if (!in_file) return ERROR;						// file data.txt not found

	while (NULL != (tok = next_token (in_file))) {	// get the next token
		Dbg ("%s", tok);
		if (NULL == list) {
			list = create_list ();
			update_token (list, tok);			// first token in list
			continue;
		}

		insert_token_into_list (&list, tok);
	}

	res = math_process_list (list);

	print_histogram (res, list);
	return 0;
}