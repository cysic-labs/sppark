// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ff/alt_bn128.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

void DumpHex(const void* data, size_t size);

extern "C"
RustError mult_pippenger(point_t* out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[])
{    
    DumpHex((const void *) &scalars[0], 32);
    DumpHex((const void *) &scalars[1], 32);
    DumpHex((const void *) &scalars[2], 32);
    printf("\n");
    DumpHex((const void *) &points[0], 64);
    printf("\n");
    DumpHex((const void *) &points[1], 64);
    printf("\n");
    DumpHex((const void *) &points[2], 64);
    printf("\n");
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}

void DumpHex(const void* data, size_t size) {
	char ascii[17];
	size_t i, j;
	ascii[16] = '\0';
	for (i = 0; i < size; ++i) {
		printf("%02X ", ((unsigned char*)data)[i]);
		if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
			ascii[i % 16] = ((unsigned char*)data)[i];
		} else {
			ascii[i % 16] = '.';
		}
		if ((i+1) % 8 == 0 || i+1 == size) {
			printf(" ");
			if ((i+1) % 16 == 0) {
				printf("|  %s \n", ascii);
			} else if (i+1 == size) {
				ascii[(i+1) % 16] = '\0';
				if ((i+1) % 16 <= 8) {
					printf(" ");
				}
				for (j = (i+1) % 16; j < 16; ++j) {
					printf("   ");
				}
				printf("|  %s \n", ascii);
			}
		}
	}
}
#endif
