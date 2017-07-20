#!/bin/bash
source /usr/local/listen/listen.sh
cd /datax/scratch/nbsearch
shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_70907_DIAG_BIRDIE_0012.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71037_DIAG_BIRDIE_0013.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71167_DIAG_BIRDIE_0014.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71297_DIAG_BIRDIE_0015.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71427_DIAG_BIRDIE_0016.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71557_DIAG_BIRDIE_0017.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71687_DIAG_BIRDIE_0018.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71817_DIAG_BIRDIE_0019.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73016_DIAG_BIRDIE_0020.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73146_DIAG_BIRDIE_0021.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73276_DIAG_BIRDIE_0022.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73406_DIAG_BIRDIE_0023.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73536_DIAG_BIRDIE_0024.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73667_DIAG_BIRDIE_0025.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_78455_DIAG_BIRDIE_0026.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_78586_DIAG_BIRDIE_0027.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_79237_DIAG_BIRDIE_0032.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_79498_DIAG_BIRDIE_0034.gpuspec.0000.fil


/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71037_DIAG_BIRDIE_0013.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_70907_DIAG_BIRDIE_0012.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71297_DIAG_BIRDIE_0015.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71167_DIAG_BIRDIE_0014.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71557_DIAG_BIRDIE_0017.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71427_DIAG_BIRDIE_0016.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71817_DIAG_BIRDIE_0019.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_71687_DIAG_BIRDIE_0018.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73146_DIAG_BIRDIE_0021.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73016_DIAG_BIRDIE_0020.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73406_DIAG_BIRDIE_0023.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73276_DIAG_BIRDIE_0022.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73667_DIAG_BIRDIE_0025.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_73536_DIAG_BIRDIE_0024.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_78586_DIAG_BIRDIE_0027.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_78455_DIAG_BIRDIE_0026.gpuspec.0000.fil

/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_79498_DIAG_BIRDIE_0034.gpuspec.0000.fil -b /datax/dibas/AGBT16B_999_127/GUPPI/BLP${NODENUM}/blc${NODENUM}_guppi_57781_79237_DIAG_BIRDIE_0032.gpuspec.0000.fil

