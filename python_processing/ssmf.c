#include <stdio.h>
#include "ssmf.h"
/*
  Spread spectrum matched filter and decimation. 
  Juha Vierinen
 */

// hardcode these to make it compile to faster code
#define NRANGE 2000
#define RESLEN 100
#define DEC 200
/* code has to be a real valued code */
int ssmf(float *meas, float *code, float *result)
{
  for(int rg=0 ; rg<NRANGE ; rg++)
  {
    for(int ti=0 ; ti<RESLEN ; ti++)
    {
      for(int di=0 ; di<DEC ; di++)
      {
	int ci=ti*DEC+di;
	// re
	result[2*(rg*RESLEN+ti)]+=meas[2*(ci+rg)]*code[ci];
	// im
	result[2*(rg*RESLEN+ti)+1]+=meas[2*(ci+rg)+1]*code[ci];
      }
    }
  }
}
