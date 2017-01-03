/*
 * Simple IDCT
 *
 * Copyright (c) 2001 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * simpleidct in C.
 */

#include "libavutil/intreadwrite.h"
#include "avcodec.h"
#include "mathops.h"
#include "simple_idct.h"

#define BIT_DEPTH 8
#include "simple_idct_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "simple_idct_template.c"

#define EXTRA_SHIFT  2
#include "simple_idct_template.c"

#undef EXTRA_SHIFT
#undef BIT_DEPTH

#define BIT_DEPTH 12
#include "simple_idct_template.c"
#undef BIT_DEPTH

/* 2x4x8 idct */

#define CN_SHIFT 12
#define C_FIX(x) ((int)((x) * (1 << CN_SHIFT) + 0.5))
#define C1 C_FIX(0.6532814824)
#define C2 C_FIX(0.2705980501)

/* row idct is multiple by 16 * sqrt(2.0), col idct4 is normalized,
   and the butterfly must be multiplied by 0.5 * sqrt(2.0) */
#define C_SHIFT (4+1+12)

// STEINAR IDCT
#define DCTSIZE 8
#define DCTSIZE2 64

// Scale factors; 1.0 / (sqrt(2.0) * cos(k * M_PI / 16.0)), except for the first which is 1.
static const double scalefac[] = {
    1.0, 0.7209598220069479, 0.765366864730180, 0.8504300947672564,
    1.0, 1.2727585805728336, 1.847759065022573, 3.6245097854115502
};

// 1D 8-point DCT.
static inline void idct1d_float(double y0, double y1, double y2, double y3, double y4, double y5, double y6, double y7, double *x)
{
    // constants
    static const double a1 = 0.7071067811865474;   // sqrt(2)
    static const double a2 = 0.5411961001461971;   // cos(3/8 pi) * sqrt(2)
    // static const double a3 = a1;
    static const double a3 = 0.7071067811865474;
    static const double a4 = 1.3065629648763766;   // cos(pi/8) * sqrt(2)
    // static const double a5 = 0.5 * (a4 - a2);
    static const double a5 = 0.3826834323650897;

    // phase 1
    const double p1_0 = y0;
    const double p1_1 = y4;
    const double p1_2 = y2;
    const double p1_3 = y6;
    const double p1_4 = y5;
    const double p1_5 = y1;
    const double p1_6 = y7;
    const double p1_7 = y3;

    // phase 2
    const double p2_0 = p1_0;
    const double p2_1 = p1_1;
    const double p2_2 = p1_2;
    const double p2_3 = p1_3;
    const double p2_4 = p1_4 - p1_7;
    const double p2_5 = p1_5 + p1_6;
    const double p2_6 = p1_5 - p1_6;
    const double p2_7 = p1_4 + p1_7;

    // phase 3
    const double p3_0 = p2_0;
    const double p3_1 = p2_1;
    const double p3_2 = p2_2 - p2_3;
    const double p3_3 = p2_2 + p2_3;
    const double p3_4 = p2_4;
    const double p3_5 = p2_5 - p2_7;
    const double p3_6 = p2_6;
    const double p3_7 = p2_5 + p2_7;

    // phase 4
    const double p4_0 = p3_0;
    const double p4_1 = p3_1;
    const double p4_2 = a1 * p3_2;
    const double p4_3 = p3_3;
    const double p4_4 = p3_4 * -a2 + (p3_4 + p3_6) * -a5;
    const double p4_5 = a3 * p3_5;
    const double p4_6 = p3_6 * a4 + (p3_4 + p3_6) * -a5;
    const double p4_7 = p3_7;

    // phase 5
    const double p5_0 = p4_0 + p4_1;
    const double p5_1 = p4_0 - p4_1;
    const double p5_2 = p4_2;
    const double p5_3 = p4_2 + p4_3;
    const double p5_4 = p4_4;
    const double p5_5 = p4_5;
    const double p5_6 = p4_6;
    const double p5_7 = p4_7;

    // phase 6
    const double p6_0 = p5_0 + p5_3;
    const double p6_1 = p5_1 + p5_2;
    const double p6_2 = p5_1 - p5_2;
    const double p6_3 = p5_0 - p5_3;
    const double p6_4 = -p5_4;
    const double p6_5 = p5_5 - p5_4;
    const double p6_6 = p5_5 + p5_6;
    const double p6_7 = p5_6 + p5_7;

    // phase 7
    x[0] = p6_0 + p6_7;
    x[1] = p6_1 + p6_6;
    x[2] = p6_2 + p6_5;
    x[3] = p6_3 + p6_4;
    x[4] = p6_3 - p6_4;
    x[5] = p6_2 - p6_5;
    x[6] = p6_1 - p6_6;
    x[7] = p6_0 - p6_7;
}

void ff_idct_float(uint8_t *dest_, int line_size, int16_t *input_)
{
    double temp[DCTSIZE2];
    double quant_table[DCTSIZE2];
    uint16_t *dest = (uint16_t *)dest_;
    int32_t *input = (int32_t *)input_;

    for (unsigned y = 0; y < DCTSIZE; ++y) {
        for (unsigned x = 0; x < DCTSIZE; ++x) {
            quant_table[y * DCTSIZE + x] = (1.0/DCTSIZE) * scalefac[x] * scalefac[y];
        }
    }

    // IDCT columns.
    for (unsigned x = 0; x < DCTSIZE; ++x) {
        idct1d_float(input[DCTSIZE * 0 + x] * quant_table[DCTSIZE * 0 + x],
                     input[DCTSIZE * 1 + x] * quant_table[DCTSIZE * 1 + x],
                     input[DCTSIZE * 2 + x] * quant_table[DCTSIZE * 2 + x],
                     input[DCTSIZE * 3 + x] * quant_table[DCTSIZE * 3 + x],
                     input[DCTSIZE * 4 + x] * quant_table[DCTSIZE * 4 + x],
                     input[DCTSIZE * 5 + x] * quant_table[DCTSIZE * 5 + x],
                     input[DCTSIZE * 6 + x] * quant_table[DCTSIZE * 6 + x],
                     input[DCTSIZE * 7 + x] * quant_table[DCTSIZE * 7 + x],
                     temp + x * DCTSIZE);
    }

    //printf("\n post idct \n");
    // IDCT rows.
    for (unsigned y = 0; y < DCTSIZE; ++y) {
        double temp2[DCTSIZE];
        idct1d_float(temp[DCTSIZE * 0 + y],
                     temp[DCTSIZE * 1 + y],
                     temp[DCTSIZE * 2 + y],
                     temp[DCTSIZE * 3 + y],
                     temp[DCTSIZE * 4 + y],
                     temp[DCTSIZE * 5 + y],
                     temp[DCTSIZE * 6 + y],
                     temp[DCTSIZE * 7 + y],
                     temp2);

        for (unsigned x = 0; x < DCTSIZE; ++x) {
            const double val = temp2[x] / 8;
            if( val > 1023)
                dest[x] = 1023;
            else if( val < 0)
                dest[x] = 0;
            else
                dest[x] = val;
            //printf("%10f ", val);

        }
        dest += line_size / 2;
        //printf("\n");
    }
    //printf("\n \n");
}
// END OF STEINAR CODE

>>>>>>> Add dev hacks
static inline void idct4col_put(uint8_t *dest, ptrdiff_t line_size, const int16_t *col)
{
    int c0, c1, c2, c3, a0, a1, a2, a3;

    a0 = col[8*0];
    a1 = col[8*2];
    a2 = col[8*4];
    a3 = col[8*6];
    c0 = ((a0 + a2) * (1 << CN_SHIFT - 1)) + (1 << (C_SHIFT - 1));
    c2 = ((a0 - a2) * (1 << CN_SHIFT - 1)) + (1 << (C_SHIFT - 1));
    c1 = a1 * C1 + a3 * C2;
    c3 = a1 * C2 - a3 * C1;
    dest[0] = av_clip_uint8((c0 + c1) >> C_SHIFT);
    dest += line_size;
    dest[0] = av_clip_uint8((c2 + c3) >> C_SHIFT);
    dest += line_size;
    dest[0] = av_clip_uint8((c2 - c3) >> C_SHIFT);
    dest += line_size;
    dest[0] = av_clip_uint8((c0 - c1) >> C_SHIFT);
}

#define BF(k) \
{\
    int a0, a1;\
    a0 = ptr[k];\
    a1 = ptr[8 + k];\
    ptr[k] = a0 + a1;\
    ptr[8 + k] = a0 - a1;\
}

/* only used by DV codec. The input must be interlaced. 128 is added
   to the pixels before clamping to avoid systematic error
   (1024*sqrt(2)) offset would be needed otherwise. */
/* XXX: I think a 1.0/sqrt(2) normalization should be needed to
   compensate the extra butterfly stage - I don't have the full DV
   specification */
void ff_simple_idct248_put(uint8_t *dest, ptrdiff_t line_size, int16_t *block)
{
    int i;
    int16_t *ptr;

    /* butterfly */
    ptr = block;
    for(i=0;i<4;i++) {
        BF(0);
        BF(1);
        BF(2);
        BF(3);
        BF(4);
        BF(5);
        BF(6);
        BF(7);
        ptr += 2 * 8;
    }

    /* IDCT8 on each line */
    for(i=0; i<8; i++) {
        idctRowCondDC_8(block + i*8, 0);
    }

    /* IDCT4 and store */
    for(i=0;i<8;i++) {
        idct4col_put(dest + i, 2 * line_size, block + i);
        idct4col_put(dest + line_size + i, 2 * line_size, block + 8 + i);
    }
}

/* 8x4 & 4x8 WMV2 IDCT */
#undef CN_SHIFT
#undef C_SHIFT
#undef C_FIX
#undef C1
#undef C2
#define CN_SHIFT 12
#define C_FIX(x) ((int)((x) * M_SQRT2 * (1 << CN_SHIFT) + 0.5))
#define C1 C_FIX(0.6532814824)
#define C2 C_FIX(0.2705980501)
#define C3 C_FIX(0.5)
#define C_SHIFT (4+1+12)
static inline void idct4col_add(uint8_t *dest, ptrdiff_t line_size, const int16_t *col)
{
    int c0, c1, c2, c3, a0, a1, a2, a3;

    a0 = col[8*0];
    a1 = col[8*1];
    a2 = col[8*2];
    a3 = col[8*3];
    c0 = (a0 + a2)*C3 + (1 << (C_SHIFT - 1));
    c2 = (a0 - a2)*C3 + (1 << (C_SHIFT - 1));
    c1 = a1 * C1 + a3 * C2;
    c3 = a1 * C2 - a3 * C1;
    dest[0] = av_clip_uint8(dest[0] + ((c0 + c1) >> C_SHIFT));
    dest += line_size;
    dest[0] = av_clip_uint8(dest[0] + ((c2 + c3) >> C_SHIFT));
    dest += line_size;
    dest[0] = av_clip_uint8(dest[0] + ((c2 - c3) >> C_SHIFT));
    dest += line_size;
    dest[0] = av_clip_uint8(dest[0] + ((c0 - c1) >> C_SHIFT));
}

#define RN_SHIFT 15
#define R_FIX(x) ((int)((x) * M_SQRT2 * (1 << RN_SHIFT) + 0.5))
#define R1 R_FIX(0.6532814824)
#define R2 R_FIX(0.2705980501)
#define R3 R_FIX(0.5)
#define R_SHIFT 11
static inline void idct4row(int16_t *row)
{
    int c0, c1, c2, c3, a0, a1, a2, a3;

    a0 = row[0];
    a1 = row[1];
    a2 = row[2];
    a3 = row[3];
    c0 = (a0 + a2)*R3 + (1 << (R_SHIFT - 1));
    c2 = (a0 - a2)*R3 + (1 << (R_SHIFT - 1));
    c1 = a1 * R1 + a3 * R2;
    c3 = a1 * R2 - a3 * R1;
    row[0]= (c0 + c1) >> R_SHIFT;
    row[1]= (c2 + c3) >> R_SHIFT;
    row[2]= (c2 - c3) >> R_SHIFT;
    row[3]= (c0 - c1) >> R_SHIFT;
}

void ff_simple_idct84_add(uint8_t *dest, ptrdiff_t line_size, int16_t *block)
{
    int i;

    /* IDCT8 on each line */
    for(i=0; i<4; i++) {
        idctRowCondDC_8(block + i*8, 0);
    }

    /* IDCT4 and store */
    for(i=0;i<8;i++) {
        idct4col_add(dest + i, line_size, block + i);
    }
}

void ff_simple_idct48_add(uint8_t *dest, ptrdiff_t line_size, int16_t *block)
{
    int i;

    /* IDCT4 on each line */
    for(i=0; i<8; i++) {
        idct4row(block + i*8);
    }

    /* IDCT8 and store */
    for(i=0; i<4; i++){
        idctSparseColAdd_8(dest + i, line_size, block + i);
    }
}

void ff_simple_idct44_add(uint8_t *dest, ptrdiff_t line_size, int16_t *block)
{
    int i;

    /* IDCT4 on each line */
    for(i=0; i<4; i++) {
        idct4row(block + i*8);
    }

    /* IDCT4 and store */
    for(i=0; i<4; i++){
        idct4col_add(dest + i, line_size, block + i);
    }
}

void ff_prores_idct(int16_t *block, const int16_t *qmat)
{
    int i;

    for (i = 0; i < 64; i++)
        block[i] *= qmat[i];

    for (i = 0; i < 8; i++)
        idctRowCondDC_extrashift_10(block + i*8, 2);

    for (i = 0; i < 8; i++) {
        block[i] += 8192;
        idctSparseCol_extrashift_10(block + i);
    }
}
