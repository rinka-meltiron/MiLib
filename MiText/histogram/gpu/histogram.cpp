#include <getopt.h>

#include <milib.h>
#include <cuda_milib.h>

// extern functions
extern void set_ab_values (all_bufs *ab);
extern ssize_t cuda_read_buffer_into_gpu (all_bufs *ab, char *f_name);
extern void create_stop_word (all_bufs *ab, FILE *fl);
extern void math_histogram (all_bufs *ab);
extern void cuda_free_everything (all_bufs *ab);
extern void histo_initialize_global_structs (all_bufs *ab, unsigned int wds);
extern void histo_time_taken (struct timeval *tvDiff, struct timeval *t2, struct timeval *t1, const char *messg);
extern void memory_compaction (all_bufs *ab, token_tracking *tt_lst, unsigned int c_size);
extern void print_token_tracking (gpu_calc *gb, token_tracking *hst);		// DEBUG

#define INPUT_FILES					256

typedef struct args_for_processing {
	bool apply_stop_words;			// stop words
	bool deamonize;					// hold and expect more files or query
	// index held on both cpu & gpu
	bool create_new_index;			// clean existing index & create new one
	// only if daemonized
	bool output;					// write to stdout or to file

	char *stop_words_file;			// filename of stop_words
	char *in_files [INPUT_FILES];	// list of input file names
	FILE *output_file;				// the actual output file
} args_for_processing;

#define no_argument					0
#define required_argument			1
#define optional_argument			2

static struct option long_options [] = {
	{"stop_words", required_argument, NULL, 's'},
	{"deamonize", optional_argument, NULL, 'D'},
	{"create_new_index", no_argument, NULL, 'c'},
	{"output", optional_argument, NULL, 'o'},
	{NULL, 0, NULL, 0}
};

// local functions
/***
static void print_math_data (math_results *mr);
static void print_math_data (math_results *mr)
{
	printf ("Tot Unique Words %u, Tot Words before %u, Tot Words after %u\n", mr -> h_tot_unique_wds, mr -> h_tot_wds_pre_sw, mr -> h_tot_wds_pst_sw);
	printf ("median %u average %u min %u max %u std_dev %f \n", mr -> median.num, mr -> avg.num, mr -> min.num, mr -> max.num, mr -> std_dev);
}
***/

static void usage (char **v);
static void usage (char **v)
{
	printf ("Usage:\nThe short form is: ./%s [-s <stopfile.txt>] [-D [-c]] [-o [outfile]] text1.txt text2.txt...\n", basename (v [0]));
	printf ("and the long form is: ./%s [--stop_words <stopfile.txt>] [--deamonize [--create_new_index]] [--output [outfile]] text1.txt text2.txt...\n", basename (v [0]));
	printf ("where:\n");
	printf ("\t-D: daemonize\n\t-c: create_new_index - goes with the -D option only\n\t-s <stop_file>: Stop words file to be read\n\t-o [outfile]: output to outfile if given or else to stdout\n");
	printf ("\tThe following functions have not been implemented:\n\t -D --daemonize\n\t--output -o\n\tmultiple input files: At this point we handle just one file");
}

static bool file_exists (const char *fname);
static bool file_exists (const char *fname)
{
	if (-1 != access (fname, F_OK)) return true;
	else return false;
}

static bool store_input_files (args_for_processing *afp, char **v, const int start, const int end);
static bool store_input_files (args_for_processing *afp, char **v, const int start, const int end)
{
	static unsigned file_loc = 0;
	bool ret = true;
	if (start == end) {
		Dbg ("Error: No input files given\nExiting...");
		ret = false;
	}
	if (end > (start + 1)) {
		Dbg ("Error: More than one input file given. Not handled\nExiting...");
		ret = false;
	}

	if (true == ret) {
		for (int i = start; i < end; i++) {
			Dbg ("to add %s to input_files at loc %u", v [i], file_loc);
			if (INPUT_FILES <= file_loc) {
				Dbg ("Max input files gotten, can't handle any more files");
				Dbg ("Error exit");
				ret = false;
				break;
			}

			if (file_exists (v [i])) {
				unsigned int len = strlen (v [i]);
				afp -> in_files [file_loc] = (char *) malloc (len + 1);
				strncpy (afp -> in_files [file_loc], v [i], len);
				Dbg ("Added %s", afp -> in_files [file_loc]);
				file_loc++;
			}
			else {
				Dbg ("Invalid file %s", v [i]);
				afp -> in_files [file_loc] = NULL;
				ret = false;
				break;
			}
		}
	}

	return ret;
}

/* Set all the option flags according to the switches specified.
 * Return the index of the first non-option argument.
 * TBD: RS_DEBUG - some Not implemented. delete the code once implementation done
 */
static void arguments_processing (args_for_processing *afp, all_bufs *ab, int argc, char **argv);
static void arguments_processing (args_for_processing *afp, all_bufs *ab, int argc, char **argv)
{
	bool ret = true;
	int c = 0;
	FILE *fl = NULL;

	if (1 == argc) {
		Dbg ("No args given\n");
		ret = false;
	}

	while (c != -1 && true == ret)
	{
		int oi = -1;		// If oi != NULL, it points the index of the long option relative to longopts.
		c = getopt_long (argc, argv, "?hs:D::co::", long_options, &oi);
		switch (c) {
			case 's':
				Dbg ("Stop File %s", optarg);

				if (true == (ret = file_exists (optarg))) {
					afp -> stop_words_file = optarg;
					fl = fopen (optarg, "r");
					if (fl) {
						create_stop_word (ab, fl);
						fclose (fl);
					}
					else {
						Dbg ("Could not open stop word file %s", optarg);
						ret = false;
					}
				}
				else {
					Dbg ("-s option needs a valid stop_words file");
				}
				break;

			case 'D':
				afp -> deamonize = true;
				{
					Dbg ("deamonize Not implemented");
					ret = false;		// TBD: because it is not implemented
				}
				break;

			case 'c':
				afp -> create_new_index = true;
				{
					Dbg ("create new index Not implemented");
					ret = false;		// TBD: because it is not implemented
				}
				break;

			case 'o':
				if (optarg) afp -> output_file = fopen ("optarg", "w");
				else afp -> output_file = stdout;

				{						// TBD: because not implemented
					if (afp -> output_file && stdout != afp -> output_file) {
						fclose (afp -> output_file);
						afp -> output_file = NULL;
					}
					Dbg ("write to output file Not implemented");
					ret = false;
				}
				break;

			case '?':
			case 'h':
				ret = false;
				break;
		}
	}

	ret = store_input_files (afp, argv, optind, argc);
	// arg validation
	if (!afp -> deamonize && afp -> create_new_index) {
		Dbg ("'create_new_index' goes with 'deamonize' option, not alone");
		ret = false;
	}

	if (false == ret) {
		usage (argv);
		exit (1);
	}
}

int main (int argc, char **argv)
{
	all_bufs		ab;
	ssize_t			chars_read = 0;		// chars read from stdio
	unsigned int	wds = 0;

	args_for_processing afp;
	memset (&afp, 0x00, sizeof (args_for_processing));

	afp.deamonize = false;			// default process and exit
	afp.create_new_index = false;	// default append to existing index
	afp.stop_words_file = NULL;		// no stop_words_file
	afp.output_file = NULL;			// no file

	set_ab_values (&ab);
	struct timeval t1, t2, tvDiff;		// time
	DEBUG_CODE (gettimeofday (&t1, NULL));

	arguments_processing (&afp, &ab, argc, argv);

	DEBUG_CODE (gettimeofday (&t2, NULL));
	DEBUG_CODE (histo_time_taken (&tvDiff, &t2, &t1, "======== Stop Words Loaded ========"));


	for (int i = 0; afp.in_files [i]; i++) {
		DEBUG_CODE (gettimeofday (&t1, NULL));
		chars_read = cuda_read_buffer_into_gpu (&ab, afp.in_files [i]);
		DEBUG_CODE (gettimeofday (&t2, NULL));
		DEBUG_CODE (histo_time_taken (&tvDiff, &t2, &t1, "======== Read next buffer ========"));

		// unsigned int i = 0;

		if (0 < chars_read) {
			printf ("*** Got %u chars: %.200s...\n", (unsigned) chars_read, ab.st_info.h_read_buf);

			DEBUG_CODE (gettimeofday (&t1, NULL));
			wds = process_next_set_of_tokens (&ab, chars_read);
			DEBUG_CODE (gettimeofday (&t2, NULL));
			DEBUG_CODE (histo_time_taken (&tvDiff, &t2, &t1, "======== Next set of tokens ========"));
			// printf ("--- pass %u done ---\n", i++);

			histo_initialize_global_structs (&ab, wds);
		}
	}

	/***
	// For debugging - to be deleted at the end.
	Dbg ("========= print_token_tracking before math_histogram =========\n");
	for (token_tracking *tmp = ab.h_ttrack; tmp; tmp = tmp -> next)
		print_token_tracking (&ab.def_gpu_calc, tmp);
	Dbg ("========= all done =========\n");
	***/

	// math_histogram (&ab);
	// if (ab.h_ttrack) print_math_data (&ab.h_math);
	cuda_free_everything (&ab);
	return 0;
}
