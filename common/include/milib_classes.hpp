/******************************************************
 * Copyright: Rinka Singh/Melt Iron
 * milib_classes.hpp
 ******************************************************/

#ifndef _MILIB_CLASSES_HPP_
#define _MILIB_CLASSES_HPP_

class buffer_mgt {
public:
	unsigned char	*buffer;			// buffer
	unsigned long	len;				// length of buffer

	buffer_mgt (unsigned char *buf, unsigned long l) {
		len = l;
		buffer = buf;
		reset ();
	}

	void mv_pointer (int moveby) {
		_bptr += moveby;
	}
	unsigned int get_pointer () {
		return _bptr;
	}
	void set_pointer (unsigned long ptr) {
		_bptr = ptr;
	}
	void append_to (FILE *from, unsigned long cp_size) {
		// code borrowed from: https://stackoverflow.com/questions/2371292/buffered-reading-from-stdin-using-fread-in-c
		fread (buffer + _bptr, cp_size, sizeof (char), from);
		mv_pointer (cp_size);
	}

	void reset () {
		memset (buffer, '\0', sizeof (char) * len);
		_bptr = 0;
	}
	~buffer_mgt () {					// destructor
		_bptr = 0;
		len = 0;
	}

protected:
	unsigned long	_bptr;				// ptr in buffer
};

class file_mgt {
public:
	FILE			*file;			// file
	unsigned long	len;			// length of file

	file_mgt () {					// constructor
		file = NULL;
		len = 0;
		_fptr = 0;
	}
	bool open (char *name) {
		file = fopen (name, "r");
		if (file) {
			fseek (file, 0L, SEEK_END);
			len = (unsigned long) ftell (file);

			char ch;
			while (!isalnum (ch = getc (file))) {
				len--;
				fseek (file, len, SEEK_SET);
			}
			len++;
			fseek (file, 0L, SEEK_SET);

			return true;
		}
		else {
			len = 0;
			_fptr = 0;

			return false;
		}
	}
	void close () {
		if (file) {
			fclose (file);
			file = NULL;
		}

		len = 0;
		_fptr = 0;
	}

	void mv_pointer (int moveby) {
		if (file) {
			fseek (file, moveby + _fptr, SEEK_SET);
			_fptr += moveby;
		}
		else {
			_fptr = 0;
		}
	}
	unsigned int get_pointer () {
		return _fptr;
	}
	void set_pointer (unsigned long ptr) {
		if (file) {
			fseek (file, ptr, SEEK_SET);
			_fptr = ptr;
		}
		else {
			_fptr = 0;
		}
	}

	char get_next_char () {
		char ch;
		if (file) {
			ch = fgetc (file);
			ungetc (ch, file);
		}
		else {
			ch = (char) 0xffff;		// error - shouldn't happen
		}

		return ch;
	}
	void read_from (char *tgt, unsigned long cp_size) {
		fread (tgt, cp_size, sizeof (char), file);
	}

	~file_mgt () {					// destructor
		if (NULL != file) {
			delete [] file;
			file = NULL;
		}
		_fptr = 0;
		len = 0;
	}

protected:
	unsigned long	_fptr;			// ptr in file
};

#endif 		// _MILIB_CLASSES_HPP_