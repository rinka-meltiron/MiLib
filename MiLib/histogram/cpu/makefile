#DEBUG =
DEBUG = TRUE
LINK_TARGET = histogram
OBJS = histogram.o
HFILES = histogram.h
MAKEFILE = makefile
REBUILDABLES = $(OBJS) $(LINK_TARGET)
CC = gcc

ifdef DEBUG
CFLAGS = -Wall -ggdb
DEFINES =
else
CFLAGS = -Wall
DEFINES = -DNDEBUG=1
endif

all: $(LINK_TARGET)

$(LINK_TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c $(HFILES) $(MAKEFILE)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ -c $<

clean:
	rm -f *~ $(REBUILDABLES) core

lint:
	cppcheck -q *c *.h