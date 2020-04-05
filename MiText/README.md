# MiText
These are the core Text Analytics Libraries from Melt Iron.  The main directories are:
* common: .h, cpp and cu files that are used across applications.
* Documentation: The various documentation.
* histogram: a text based histogram
* MiSQL: a proto-DB that is GPU based
* tools: various scripts and other tools

## Histogram directory
This is a histogram of a text file.  Essentially:
* Apply stop words (optional)
* Sort words of an inputted text file
* Merges words and tracks the frequency of the words
* Stores it on the GPU.

We also do Math analysis: Mean, Average, Std Dev.

### Setup
The environment setup is as described on the [README.md page on MiLib](https://github.com/rinka-meltiron/MiLib#setup)

### Update and Current status
5-Apr-20:
* Histogram works on the GPU.
* UX for reading multiple files from command-line is done.
* However the MiText Engine handles just one file.  TBD in the next release.
* Download and compile issues of the build are fixed
* Readme.MD updated.
* New files read_multi_file.tdd, directories obj_dir are checked in.

### Compiling
* clone MiLib and create a branch in your current directory ./
* The relevant makefiles are ```./common/lib/Makefile``` and ```./histogram/gpu/Makefile```
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
  $ ./histogram -s stop_file.txt file.txt file1.txt		# take stopwords as input
						# from stop_file.txt, apply that to file.txt and file1.txt
						# & then create a histogram
```
This results in a sorted histogram on the GPU.<br>
We do not output anything at this point as the assumption is that we will hold the data on the GPU and send queries there.

the various alternative parameters are.  These have yet to be implemented
```
  $ ./histogram -s stop_file.txt -o output.txt file.txt...		# write output to output.txt
  $ ./histogram -s stop_file.txt -o output.txt -D file.txt...	# histogram as daemon on CPU & GPU.
  $ ./histogram -s stop_file.txt -o output.txt -D -c file.txt...	# histogram as daemon on both CPU & GPU,
								# wipe out existing histogram & recreate
  $ ./histogram -s stop_file.txt -o output.txt -D -c f1.txt f2.txt f3.txt ...
								# pass input files: f1.txt f2.txt f3.txt
								# hold histogram as daemon on CPU & GPU,
								# wipe out existing histogram & recreate
```
where:<br>
```
-s parameter to pass in a stop_words_file file.
stop_words_file is a list of stop-words in text format (we assume there are no duplicates).
-o output.file writes the output to a file.  If -o is specified without a filename, then the output is written to stdout<br>
-D daemonizes the histogram. You can then use histogram to pass it further files and commands
-c wipes out and creates a new index.  The previous histogram (if it exists because of the -D parameter) is wiped out and a new index is created with the files passed
file.txt, f1.txt, f2.txt, f3.txt etc. are text files.
```

## Inverted Index
is TBD.  We plan to extend the histogram stored by:
* Enabling reading multiple text files
* applying stop words to these
* Merging them with the existing text histogram - including storing file-names.  Alternatively we can also store the memory location of each word with file-name allowing a fine grained indexing for each word but this will be pretty expensive.
* Allowing querying for a specific word and returning the number of words and file-name.
