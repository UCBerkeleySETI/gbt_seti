#!/bin/bash
source /usr/local/listen/listen.sh
cd /datax/scratch/nbsearch/mltest
rm /datax/scratch/nbsearch/mltest/*

shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}
#mkdir /datax/scratch/mlplots/${NODENUM}


${NODENUM}
DIR=/datax/dibas/AGBT17A_999_05/GUPPI/BLP${NODENUM}/blc${NODENUM}
#blc17_guppi_57798_55151_DIAG_MASER_A_0006.gpuspec.0000.fil
#-rw-r--r-- 1 root root  4294967694 Feb 14 15:58 blc17_guppi_57798_55702_DIAG_MASER_A_OFF_0007.gpuspec.0000.fil
#/datax/dibas/AGBT17A_999_05/GUPPI/BLP17/

echo ${DIR}

/usr/local/listen/bin/filterbanksearch -a ${DIR}_guppi_57798_55151_DIAG_MASER_A_0006.gpuspec.0002.fil -b ${DIR}_guppi_57798_55702_DIAG_MASER_A_OFF_0007.gpuspec.0002.fil
#/usr/local/listen/bin/filterbanksearch -a ${DIR}_guppi_57798_55151_DIAG_MASER_A_0006.gpuspec.0000.fil -b ${DIR}_guppi_57798_55702_DIAG_MASER_A_OFF_0007.gpuspec.0000.fil
#/usr/local/listen/bin/filterbanksearch -a ${DIR}_guppi_57798_55151_DIAG_MASER_A_0006.gpuspec.0001.fil -b ${DIR}_guppi_57798_55702_DIAG_MASER_A_OFF_0007.gpuspec.0001.fil


find /datax/scratch/nbsearch/mltest -name \*fits -exec /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -r "fits_plot('{}',0.1,0.9,10,'{}','dark',0); quit" \;

#cp /datax/scratch/nbsearch/mltest/*eps /datax/scratch/plots/${NODENUM}
