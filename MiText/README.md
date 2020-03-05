# MiText
These are the core Text Analytics Libraries from Melt Iron.  The main directories are:
* common: .h, cpp and cu files that are used across applications.
* Documentation: The various documentation.  Currently not updated.
* histogram: a text based histogram
* MiSQL: a proto-DB that is GPU based
* tools: various scripts and other tools

## Histogram directory
This is a histogram of a text file.  Essentially:
* Apply stop words (optional)
* Sort words of an inputted text file
* Merges words and tracks the frequency of the words
* Stores it on the GPU.

We also do Math analysis: Mean, Average, Std Dev however this has not been tested completely since this is not critical.
The function has been commented out but can be enabled.  At this point it does the Math calculation sequentially and will impact performance.

### Setup
The environment setup is as described on the [README.md page on MiLib](https://github.com/rinka-meltiron/MiLib)

### Compiling
* clone MiLib as a branch in your current directory ./
* The relevant makefiles are ```MiLib/common/lib/Makefile``` and ```MiLib/histogram/gpu/Makefile```
* These are currently set in debug mode.  For production version, modify the variable *dbg=1* in each of the Makefiles to *dbg=0*.

Compilation consists of two steps:

1. compile the common files first
  This will create the library histo.a  The commands are:

```
  $ cd MiLib/common/lib         # contains common files.
  $ make                        # Makefile here creates histo.a
```

2. Compile the histogram files
```
  $ cd ../../histogram/gpu/     # the histogram files
  $ make                        # creates ./histogram
```

### Running
the command is:
```
  $ cd ./MiLib/histogram/gpu/			# cd to the appropriate directory
  $ ./histogram file.txt			# takes file.txt as input and creates a histogram
						# on both the GPU and the CPU
  $ ./histogram -s stop_file.txt file.txt	# take stopwords as input from
						# stop_file.txt, applies that to file.txt and then
						# create a histogram
```
This results in a sorted histogram on the GPU.<br>
We do not output anything at this point as the assumption is that we will hold the data on the GPU and send queries there.

the various alternative parameters are.  These have yet to be implemented
```
  $ ./histogram -s stop_file.txt -o output.txt file.txt		# write output to output.txt
  $ ./histogram -s stop_file.txt -o output.txt -D file.txt	# histogram as daemon on CPU & GPU.
  $ ./histogram -s stop_file.txt -o output.txt -D -c file.txt	# histogram as daemon on both CPU & GPU.
								# Also, wipe out existing histogram & create anew
  $ ./histogram -s stop_file.txt -o output.txt -D -c f1.txt f2.txt f3.txt ...
								# pass multiple input files f1.txt, f2.txt, f3.txt
								# hold histogram as daemon on CPU & GPU.
								# in addition, wipe out existing histogram and
								# create a new one with 'f1.txt'
```
where:<br>
stop_words_file is a list of stop-words in text format (we assume there are no duplicates).<br>
file.txt, f1.txt, f2.txt, f3.txt etc. are text files.<br>

## Inverted Index
will be implemented as a subsequent step.  We plan to extend the histogram stored by:
* Enabling reading multiple text files
* applying stop words to these
* Merging them with the existing text histogram - including storing file-names.  Alternatively we can also store the memory location of each word with file-name allowing a fine grained indexing for each word but this will be pretty expensive.
* Allowing querying for a specific word and returning the number of words and file-name.
