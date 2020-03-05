# Setup
The environment setup is as follows:
* The MiText build has been tested with CUDA 7.5 on a local machine with an nVidia Quadro 2000 card.  Some modifications are needed to the Makefile to compile it under CUDA 8.0.
* This has been tested on Ubuntu 14.4 and 16.6.
* You should either have a Desktop/laptop with a CUDA capable nVidia card or access to a CUDA instance on the cloud (AWS, Google,Azure).
* The CUDA card and RAM on the instance should be enough to load the data completely on the card.  At this point, we do not handle the situation where the text data is larger than the CUDA memory.
* You should also ensure that your instance/machine can compile and run CUDA 7.5 code.

# MiText
These are the core Text Analytics Libraries from Melt Iron.  The main directories are:
* common: .h, cpp and cu files that are used across applications.
* Documentation: The various documentation.  Currently not updated.
* histogram: a text based histogram
* tools: various scripts and other tools

# MiSQL
Is a proto DB that handled one table and ran a simple pattern matching query on it.  It has not been updated with the latest include files.
