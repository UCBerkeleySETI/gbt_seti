#!/bin/bash
shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]}
NODENUM=${myarray[0]##*( )}

echo /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71427_DIAG_BIRDIE_0016.gpuspec.0000.fil

