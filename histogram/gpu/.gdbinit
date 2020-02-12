# set args -s data/all/stop-word-merged.txt < data/all/1line.txt
# set args -s data/all/stop-word-merged.txt < data/all/2line.txt
# set args -s data/all/stop-word-merged.txt < data/all/vlfile.txt
# set args -s data/all/stop-word-merged.txt < data/all/Dan\ Simmons\ -\ Hyperion.txt
# set args -s data/all/stop-word-merged.txt < data/all/1para.txt
set args -s data/stop-wd-10.txt < data/all/1para.txt
# set args -s data/stop-wd-2.txt < data/all/1para.txt

# set args -s data/sw.txt < data/all/the.txt
# set args -s data/sw.txt < data/all/1line.txt
# set args -s data/all/stop-word-merged.txt < data/all/3line.txt
# set args -s data/all/stop-word-merged.txt < data/all/1para.txt
# set args -s data/stop-wd-10.txt < data/all/3line.txt
# set args -s data/sw.txt < data/all/3line.txt
# set args -s data/sw.txt < data/all/1para.txt
# set args < data/all/1line.txt
# set args < data/all/the.txt
# tb main
# b milib_gpu_sort_merge_histo_wds
# b sort_merge_histo_cross
# b cuda_create_stop_word_lst
# b apply_stop_words
# b memory_compaction
# b math_histogram
# b histo_math_cp_sort_Dlist

# b copy_scratchpad_to_multi_tt

# set args -s data/sw.txt < data/all/3the-one_line.txt
# set args < data/all/1line.txt
# set args -s data/sw.txt < data/all/2line.txt
# b milib_gpu_sort_merge_histo_wds(all_bufs*, bool)
# b cuda_convert_to_words(long, all_bufs*)

# set check range auto
set check type on
# set complaints 10000
set demangle-style gnu
set mem inaccessible-by-default
# set verbose on
# ========================
