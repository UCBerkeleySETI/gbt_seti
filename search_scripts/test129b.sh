#!/bin/bash
source /usr/local/listen/listen.sh
cd /datax/scratch/nbsearch/boyajian

shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}
mkdir /home/siemion/plots/${NODENUM}


${NODENUM}
DIR=/datax/DIAG_Boyajian_4/AGBT16B_999_129/GUPPI/BLP${NODENUM}/blc${NODENUM}

echo ${DIR}

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_58531_maserA_0012.gpuspec.0002.fil -b ${DIR}_guppi_57783_58731_KIC8462852_0013.gpuspec.0002.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_63624_maserA_0027.gpuspec.0002.fil -b ${DIR}_guppi_57783_63790_KIC8462852_0028.gpuspec.0002.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_68593_maserA_0060.gpuspec.0002.fil -b ${DIR}_guppi_57783_68431_KIC8462852_0059.gpuspec.0002.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_69293_maserA_0063.gpuspec.0002.fil -b ${DIR}_guppi_57783_69454_KIC8462852_0064.gpuspec.0002.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_73963_maserA_0096.gpuspec.0002.fil -b ${DIR}_guppi_57783_73803_KIC8462852_0095.gpuspec.0002.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_74655_maserA_0099.gpuspec.0002.fil -b ${DIR}_guppi_57783_74814_KIC8462852_0100.gpuspec.0002.fil

#find /datax/scratch/nbsearch/boyajian -name \*fits -exec /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -r "fits_plot('{}',0.1,0.9,10,'{}','dark',0); quit" \;

#cp /datax/scratch/nbsearch/boyajian/*eps /home/siemion/plots/${NODENUM}
