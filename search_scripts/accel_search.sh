#$ -N accel_search
#$ -cwd
#$ -S /bin/bash
#$ -t 1-32

. /etc/profile.d/modules.sh
module load gcc

export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot

files=(`ls B1839+56_8bit.fil/*.dat`)
numFiles=$((${#files[*]} / 32))

echo "Starting Accel Search: `date`"
ls ${files[@]:$((($SGE_TASK_ID-1)*$numFiles)):$numFiles} | xargs -n 1 $PRESTO/bin/accelsearch -zmax 0 -harmpolish > SearchOutput.txt
echo "Finished Accel Search: `date`"
