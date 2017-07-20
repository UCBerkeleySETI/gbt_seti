#!/bin/bash
source /usr/local/listen/listen.sh
source /home/siemion/sw/dev/gbt_seti/src/setkeys.sh
source /home/siemion/sw/google-cloud-sdk/path.bash.inc
cd $1


shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}
mkdir /home/siemion/plots/${NODENUM}

ON_FILE=$2
OFF_FILE=$3
THRESH=$4
OBSID=$5

#empty bucket: gsutil -m rm gs://setidata/*
#/home/siemion/gpu_code/pubsub.sh s --id=MJD_5000.500000

DIR=$1/GUPPI/BLP${NODENUM}/blc${NODENUM}

rm -rf /datax/scratch/real_time_temp/*
mkdir /datax/scratch/real_time_temp/raw

/home/siemion/gpu_code/pubsub.sh r --node=blc${NODENUM} --status=1 --id=${OBSID}

cd ${1}/GUPPI/BLP${NODENUM}
numactl --cpunodebind 1 /home/obs/bin/gpuspec_wrapper ${ON_FILE}
numactl --cpunodebind 1 /home/obs/bin/gpuspec_wrapper ${OFF_FILE}


echo ${DIR}
echo ${DIR}_${ON_FILE}
echo ${DIR}_${OFF_FILE}
#guppi_57856_71489_DIAG_W3OH_OFF_0044.gpuspec.0002.fil
time /home/siemion/sw/dev/gbt_seti/src/filterbanksearchdoppler -a ${DIR}_${ON_FILE}.gpuspec.0000.fil -b ${DIR}_${OFF_FILE}.gpuspec.0000.fil -z ${THRESH} -s setidata -i ${OBSID}
time /home/siemion/sw/dev/gbt_seti/src/filterbanksearchdoppler -a ${DIR}_${ON_FILE}.gpuspec.0002.fil -b ${DIR}_${OFF_FILE}.gpuspec.0002.fil -z ${THRESH} -s setidata -i ${OBSID}
cd /datax/scratch/real_time_temp
python /home/siemion/fits_to_json.py -d /datax/scratch/real_time_temp
cd /datax/scratch/real_time_temp/processed/img
gsutil -m cp *.png gs://setidata
/home/siemion/gpu_code/pubsub.sh r --node=blc${NODENUM} --status=2 --id=${OBSID}

#/home/siemion/gpu_code/pubsub.sh s --id=MJD_5000.500000


#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_58531_maserA_0012.gpuspec.0002.fil -b ${DIR}_guppi_57783_58731_KIC8462852_0013.gpuspec.0002.fil

#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_63624_maserA_0027.gpuspec.0002.fil -b ${DIR}_guppi_57783_63790_KIC8462852_0028.gpuspec.0002.fil

#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_68593_maserA_0060.gpuspec.0002.fil -b ${DIR}_guppi_57783_68431_KIC8462852_0059.gpuspec.0002.fil

#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_69293_maserA_0063.gpuspec.0002.fil -b ${DIR}_guppi_57783_69454_KIC8462852_0064.gpuspec.0002.fil

#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_73963_maserA_0096.gpuspec.0002.fil -b ${DIR}_guppi_57783_73803_KIC8462852_0095.gpuspec.0002.fil

#/home/siemion/sw/dev/gbt_seti/src/filterbanksearch -a ${DIR}_guppi_57783_74655_maserA_0099.gpuspec.0002.fil -b ${DIR}_guppi_57783_74814_KIC8462852_0100.gpuspec.0002.fil

#find /datax/scratch/nbsearch/boyajian -name \*fits -exec /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -r "fits_plot('{}',0.1,0.9,10,'{}','dark',0); quit" \;

#cp /datax/scratch/nbsearch/boyajian/*eps /home/siemion/plots/${NODENUM}
