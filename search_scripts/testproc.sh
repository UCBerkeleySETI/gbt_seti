#!/bin/bash
source /usr/local/listen/listen.sh
source /usr/local/listen/listen.sh
find /datax/dibas -name \*DIAG_ZENITH\*0002.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0002 \;
find /datax/dibas -name \*DIAG_ZENITH\*0003.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0003 \;
find /datax/dibas -name \*DIAG_ZENITH\*0004.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0004 \;
find /datax/dibas -name \*DIAG_ZENITH\*0005.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0005 \;
find /datax/dibas -name \*DIAG_ZENITH\*0006.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0006 \;
find /datax/dibas -name \*DIAG_ZENITH\*0007.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0007 \;
find /datax/dibas -name \*DIAG_ZENITH\*0008.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0008 \; 
find /datax/dibas -name \*DIAG_ZENITH\*0009.0000.raw -execdir /usr/local/listen/bin/gpuspec2 -i {} -b 8 -B 2 -f 1048576 -t 4 -o /datax/scratch/nbsearch/B0009 \;

