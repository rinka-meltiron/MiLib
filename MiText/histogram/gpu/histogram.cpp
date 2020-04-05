/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * histogram.cpp
 ******************************************************/
#include <getopt.h>
#include <milib.h>
#include <cuda_milib.h>
#include <milib_classes.hpp>

using namespace std;
// extern functions
extern void set_ab_values (all_bufs *ab);
extern void cuda_read_buffer_into_gpu (all_bufs *ab, unsigned chars_read);
extern void create_stop_word (all_bufs *ab, FILE *fl);
extern void math_histogram (all_bufs *ab);
extern void cuda_free_everything (all_bufs *ab);
extern void histo_initialize_global_structs (all_bufs *ab, unsigned int wds);
extern void histo_time_taken (struct timeval *tvDiff, struct timeval *t2, struct timeval *t1, const char *messg);
extern void memory_compaction (all_bufs *ab, token_tracking *tt_lst, unsigned int c_size);
extern void print_token_tracking (gpu_calc *gb, token_tracking *hst);		// DEBUG

#define INPUT_FILES					256

// TBD: RS_DEBUG the INPUT_FILES is limited to 256 files.
// We need to make it extensible.
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

// store the last word
static bool push_last_word (buffer_mgt& last_wd, buffer_mgt& buf, file_mgt& file);
static bool push_last_word (buffer_mgt& last_wd, buffer_mgt& buf, file_mgt& file)
{
	if (isalnum (file.get_next_char ())) {	// push the word
		unsigned wd_sz = 0;
		unsigned buf_loc;

		if (0 == buf.get_pointer ()) return 0;	// error
		for (buf_loc = buf.get_pointer () - 1; isalnum (buf.buffer [buf_loc]) && buf_loc > 0; buf_loc--) wd_sz++;
		if (0 == buf_loc) return false;

		buf_loc++;

		strncpy ((char *) last_wd.buffer, (char *) (buf.buffer + buf_loc), wd_sz);
		last_wd.buffer [wd_sz] = ' ';
		memset (buf.buffer + buf_loc, ' ', wd_sz * sizeof (char));
		buf.set_pointer (buf_loc);			// reset the last wd in buffer
		last_wd.set_pointer (wd_sz);		// got word from buffer
	}										// else do nothing

	return true;
}

// fb_diff calculation
static int calculate_diff (buffer_mgt& buf, file_mgt& file);
static int calculate_diff (buffer_mgt& buf, file_mgt& file)
{
	return ((file.len - file.get_pointer ()) - (buf.len - buf.get_pointer ()));
}

// last word to buffer
static unsigned pop_last_word (buffer_mgt& buf, buffer_mgt& last_word);
static unsigned pop_last_word (buffer_mgt& buf, buffer_mgt& last_word)
{
	unsigned wd_len = last_word.get_pointer ();
	unsigned ret = 0;
	if (wd_len) {
		assert (wd_len + buf.get_pointer () <= buf.len);

		strncpy ((char *) (buf.buffer + buf.get_pointer ()), (char *) last_word.buffer, wd_len);
		memset (last_word.buffer, ' ', sizeof (char) * wd_len + 1);
		last_word.set_pointer (0);

		buf.mv_pointer (wd_len);
		ret = buf.len - buf.get_pointer ();
	}

	return ret;
}

// read next file into buffer till buffer full or last file is done
// we assume: last word in file is not truncated and continued in next file
static bool read_files_fill_buffer (buffer_mgt& buf, char **in_files);
static bool read_files_fill_buffer (buffer_mgt& buf, char **in_files)
{
	static unsigned char wd_buf [STAT_MAX_WORD_SIZE + 2];
	static buffer_mgt last_word (wd_buf, STAT_MAX_WORD_SIZE + 2);	// last word
	pop_last_word (buf, last_word);

	while (true) {
		static int file_loc = 0;				// at least 1 file is passed
		static bool isLastFile = false;
		static file_mgt fm;

		if (NULL == fm.file) {	        		// start of file
			if (!fm.open (in_files [file_loc++])) {
				if (false == isLastFile) {
					isLastFile = true;
					return true;
				}
				else {
					isLastFile = false;
					buf.reset ();
					last_word.reset ();
					return false;
				}
			}
		}

		// fb_diff is actual difference: file == buf:0, file < buf:-, file > buf:+
		int fb_diff = calculate_diff (buf, fm);

		if (fb_diff > 0) {						// file larger
			unsigned buf_len_to_cp = buf.len - buf.get_pointer ();
			buf.append_to (fm.file, buf_len_to_cp);
			fm.mv_pointer (buf_len_to_cp);

			if (false == push_last_word (last_word, buf, fm)) {
				printf ("error, word too large to be pushed\n");
				exit (1);
			}
			break;
		}
		else {									// file less or equal
			unsigned flle_len_to_cp = fm.len - fm.get_pointer ();
			buf.append_to (fm.file, flle_len_to_cp);
			fm.close ();						// EOF - next file

			if (fb_diff < 0) {
				buf.buffer [buf.get_pointer ()] = ' ';
				buf.mv_pointer (1);
			}
			else {								// fb_diff == 0
				break;
			}
		}

		// RS_DEBUG - remove later.
		if (1 == file_loc) {
			return true;				// since we handle just one buffer
		}
		else {
			Dbg ("We don't handle more than one file now");
			Dbg ("Exiting");
			exit (1);
		}
	}


	return true;
}

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
				afp -> in_files [file_loc] [len] = 0x00;
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

				afp -> apply_stop_words = true;
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
	if (true == afp -> apply_stop_words && NULL == afp -> stop_words_file) {
		Dbg ("Valid stop word text file not given");
		ret = false;
	}
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
	struct timeval	t1, t2, tvDiff;		// time
	DEBUG_CODE (gettimeofday (&t1, NULL));

	args_for_processing afp;
	memset (&afp, 0x00, sizeof (args_for_processing));

	afp.deamonize = false;			// default process and exit
	afp.create_new_index = false;	// default append to existing index
	afp.stop_words_file = NULL;		// no stop_words_file
	afp.output_file = NULL;			// no file

	all_bufs		ab;
	set_ab_values (&ab);
	buffer_mgt		buffer (ab.st_info.h_read_buf, \
							ab.st_info.bufsiz * sizeof (unsigned char));
	arguments_processing (&afp, &ab, argc, argv);

	DEBUG_CODE (gettimeofday (&t2, NULL));
	DEBUG_CODE (histo_time_taken (&tvDiff, &t2, &t1, "======== Stop Words Loaded ========"));

	while (true == read_files_fill_buffer (buffer, afp.in_files)) {
		ssize_t		chars_read = buffer.get_pointer ();		// chars in buffer
		cuda_read_buffer_into_gpu (&ab, chars_read);
		if (0 < chars_read) {
			unsigned int	wds = 0;
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
	// For debugging
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
