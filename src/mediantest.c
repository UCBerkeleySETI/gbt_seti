#include <stdio.h>
#include "median.h"


int main()
{

float vars[6];
vars[0] = 1;
vars[1] = 2;
vars[2] = 3;
vars[3] = 4;
vars[4] = 3;
vars[5] = 6;

printf("%f\n", median(vars, 6));


}