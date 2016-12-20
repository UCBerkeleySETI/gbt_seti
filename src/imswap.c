#include "imswap.h"

/* IMSWAP4 -- Reverse bytes of Integer*4 or Real*4 vector in place */
void imswap4 (char *string, int nbytes) 
{

/* string Address of Integer*4 or Real*4 vector */
/* bytes Number of bytes to reverse */
    char *sbyte, *slast;
    char temp0, temp1, temp2, temp3;
    slast = string + nbytes;
    sbyte = string;
    while (sbyte < slast) {
        temp3 = sbyte[0];
        temp2 = sbyte[1];
        temp1 = sbyte[2];
        temp0 = sbyte[3];
        sbyte[0] = temp0;
        sbyte[1] = temp1;
        sbyte[2] = temp2;
        sbyte[3] = temp3;
        sbyte = sbyte + 4;
        }

    return;
}
