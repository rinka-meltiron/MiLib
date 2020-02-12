#include <unistd.h>

#include <milib.h>
#include <cuda_milib.h>

// extern functions
extern void set_ab_values (all_bufs *ab);
extern ssize_t cuda_read_buffer_into_gpu (all_bufs *ab);
extern void create_stop_word (all_bufs *ab, FILE *fl);
extern void math_histogram (all_bufs *ab);
extern void cuda_free_everything (all_bufs *ab);
extern void histo_initialize_global_structs (all_bufs *ab, unsigned int wds);
extern void histo_time_taken (struct timeval *tvDiff, struct timeval *t2, struct timeval *t1, const char *messg);
extern void memory_compaction (all_bufs *ab, token_tracking *tt_lst, unsigned int c_size);
extern void print_token_tracking (gpu_calc *gb, token_tracking *hst);		// DEBUG

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
	printf ("Usage:\n\t./%s < text_file.txt\n", basename (v [0]));
	printf ("\t./%s -s stopfile.txt < text_file.txt\n\n", basename (v [0]));
}

static void arguments_processing (all_bufs *ab, int c, char **v);
static void arguments_processing (all_bufs *ab, int c, char **v)
{
	int			d;
	const char *short_opts	= "h?s:";
	FILE *fl				= NULL;

	while ((d = getopt (c, v, short_opts)) != -1) {
		switch (d) {
			case 's' :
				Dbg ("Stop File %s", optarg);
				fl = fopen (optarg, "r");
				check_mem (fl);
				create_stop_word (ab, fl);
				fclose (fl);
				return;

			case 'h' :
			case '?' :
				usage (v);
				break;

			default  :
				usage (v);
				return;
		}
		break;
	}

	if (-1 == d) return;

	error:
	usage (v);
	exit (1);
}

int main (int argc, char **argv)
{
	all_bufs		ab;
	ssize_t			chars_read = 0;		// chars read from stdio
	unsigned int	wds = 0;

	set_ab_values (&ab);
	struct timeval t1, t2, tvDiff;		// time
	DEBUG_CODE (gettimeofday (&t1, NULL));
	arguments_processing (&ab, argc, argv);
	DEBUG_CODE (gettimeofday (&t2, NULL));
	DEBUG_CODE (histo_time_taken (&tvDiff, &t2, &t1, "======== Stop Words Loaded ========"));

	while (!feof (stdin)) {
		DEBUG_CODE (gettimeofday (&t1, NULL));
		chars_read = cuda_read_buffer_into_gpu (&ab);
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
