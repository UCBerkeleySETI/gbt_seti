#!/bin/bash
source /usr/local/listen/listen.sh
cd /datax/scratch/nbsearch/boyajian
shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}


DIR=/datax/DIAG_Boyajian_4/AGBT16B_999_129/GUPPI/BLP${NODENUM}/blc${NODENUM}

echo ${DIR}

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_58531_maserA_0012.gpuspec.0002.fil -b ${DIR}_guppi_57783_57054_j2015_0008.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_63624_maserA_0027.gpuspec.0002.fil -b ${DIR}_guppi_57783_57491_j2015_0010.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_68593_maserA_0060.gpuspec.0002.fil -b ${DIR}_guppi_57783_58213_j2015_0011.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_69293_maserA_0063.gpuspec.0002.fil -b ${DIR}_guppi_57783_63310_j2015_0026.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_73963_maserA_0096.gpuspec.0002.fil -b ${DIR}_guppi_57783_68766_j2015_0061.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_74655_maserA_0099.gpuspec.0002.fil -b ${DIR}_guppi_57783_69122_j2015_0062.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_79885_maserA_0133.gpuspec.0002.fil -b ${DIR}_guppi_57783_74123_j2015_0097.gpuspec.0002.fil
/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_80017_maserA_0134.gpuspec.0002.fil -b ${DIR}_guppi_57783_74495_j2015_0098.gpuspec.0002.fil

find /datax/scratch/nbsearch/boyajian -name \*fits -exec /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -r "fits_plot('{}',0.1,0.9,10,'{}','dark',0); quit" \;



