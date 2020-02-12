# MiLib
These are the core Analytics Libraries from Melt Iron

## Histogram
The following creates a histogram of a text file.  Essentially:
* Apply stop words (if desired)
* Sorts words of an inputted text file
* Merges words and tracks the number of repetitions
* Stores it on the GPU.

We also do Math analysis: Mean, Average, Std Dev however this has not been tested completely since this is not critical.
The function has been commented out but can be enabled.  At this point it does the Math calculation sequentially and will impact performance.

### Setup

This build has been tested with CUDA 7.5 on a local machine with an nVidia Quadro 2000 card.  It will require some modifications to the Makefile to compile under CUDA 8.0.

You should either have a CUDA capable nVidia card on your machine or access to a CUDA instance on the cloud (AWS, Google,Azure).
You should also ensure that you can compile and run CUDA code on this machine under CUDA 7.5.

### Compiling
We assume you have cloned MiLib in the current directory ./
The makefiles are MiLib/common/lib/Makefile and MiLib/histogram/gpu/Makefile

These are currently set to output debug executables.  Should you want to create optimized production version, you will find the variable dbg in each of these files set to 1.  Next to that is the same variable set to 0 and commented out.  Comment out dbg=1 and enable dbg=0 for the production version.

Compilation consists of two steps:

1. compile the common files first
  This will create the library histo.a  You would do that by:
  
  $ cd MiLib/common/lib         # contains the common files.  you can compile as a common user (don't need to use sudo)
  
  $ make                        # Makefile here creates histo.a

2. Compile the histogram files

  $ cd ../../histogram/gpu/     # gets you to the directory containing the histogram files

  $ make                        # Makefile creates the executable ./histogram

### Running
you should be in the directory MiLib/histogram/gpu/

the command is:

$ ./histogram [-s <stop_words_file>] < text_file

where:
stop_words_file is a list of stop-words in text format (we assume there are no duplicates).
text_file is a text file containing words.  We have tested this to a max of 2.5 MB.

This results in a sorted histogram on the GPU.
We do not output anything at this point as the assumption is that we will hold the data on the GPU and send queries there.

## Inverted Index
We plan to extend the histogram stored by:
* Enabling reading multiple text files
* applying stop words to these
* Merging them with the existing text histogram - including storing file-names.  Alternatively we can also store the memory location of each word with file-name allowing a fine grained indexing for each word but this will be pretty expensive.
* Allowing querying for a specific word and returning the number of words and file-name.

# MiSQL
Was a proto DB that handled one table and ran a simple pattern matching query on it.  It has not been tested and may not run.  I don't plan to update this unless there's a clear need.
