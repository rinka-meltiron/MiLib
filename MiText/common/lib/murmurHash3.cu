//-----------------------------------------------------------------------------
/**
 * 32-bit Murmur3 hash.
 *
 * @param data      source data
 * @param nbytes    size of data
 *
 * @return 32-bit unsigned hash value.
 *
 * @code
 *  uint32_t hashval = Murmurhash3_32((void*)"hello", 5);
 * @endcode
 *
 * @code
 *  MurmurHash3 was created by Austin Appleby  in 2008. The initial
 *  implementation was published in C++ and placed in the public.
 *    https://sites.google.com/site/murmurhash/
 *  Seungyoung Kim has ported its implementation into C language
 *  in 2012 and published it as a part of qLibc component: https://github.com/wolkykim/qlibc
 * @endcode
 */

#include <stdint.h>
#include <cuda.h>

#include <cuda_milib.h>

__device__ uint32_t MurmurHash3_32 (const unsigned char *data, const unsigned int nbytes);

__device__ uint32_t MurmurHash3_32 (const unsigned char *data, const unsigned int nbytes)
{
	// CudaDbgPrn ("data %s, nb %u", data, (unsigned) nbytes);
	if (data == NULL || nbytes == 0) return 0;

	const int nblocks = (int) (nbytes / 4);

	uint32_t k, h = (uint32_t) 0;
	for (int i = 0; i < nblocks; i++) {
		k = (uint32_t) *(data + i);

		k *= (uint32_t) 0xcc9e2d51;
		k =  (uint32_t) ((k << 15) | (k >> 17));	// 17 = 32 -15
		k *= (uint32_t) 0x1b873593;

		h ^= (uint32_t) k;
		h =  (uint32_t) ((h << 13) | (h >> 19));	// 19 = 32 -13
		h = (uint32_t) ((h * 5) + 0xe6546b64);
	}

	k = (uint32_t) 0;
	k ^= (uint32_t) ((uint8_t) data [nbytes - 1]);
	k *= (uint32_t) 0xcc9e2d51;
	k =  (uint32_t) ((k << 15) | (k >> (32 - 15)));
	k *= (uint32_t) 0x1b873593;
	h ^= (uint32_t) k;

	h ^= (uint32_t) nbytes;

	h ^= (uint32_t) (h >> 16);
	h *= (uint32_t) 0x85ebca6b;
	h ^= (uint32_t) (h >> 13);
	h *= (uint32_t) 0xc2b2ae35;
	h ^= (uint32_t) (h >> 16);

	// CudaDbgPrn ("%s: %u", data, (unsigned) h);
	return h;
}
