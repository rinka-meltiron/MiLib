### Setup
The environment setup is as follows:
* This build has been tested with CUDA 7.5 on a local machine with an nVidia Quadro 2000 card.  Some modifications are needed to the Makefile to compile it under CUDA 8.0.
* This has been tested on Ubuntu 14.4 and 16.6.
* You should either have a Desktop/laptop with a CUDA capable nVidia card or access to a CUDA instance on the cloud (AWS, Google,Azure).
* The CUDA card and RAM on the instance should be enough to load the data completely on the card.  At this point, we do not handle the situation where the text data is larger than the CUDA memory.
* You should also ensure that your instance/machine can compile and run CUDA 7.5 code.

# MiLib
These are the core Analytics Libraries from Melt Iron.  The main directories are:
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

This will be extended to provide inverted index functionality

# MiSQL
Was a proto DB that handled one table and ran a simple pattern matching query on it.  It has not been tested and may not run.  I don't plan to update this unless there's a clear need.
