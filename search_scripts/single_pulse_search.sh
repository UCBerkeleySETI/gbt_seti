#$ -N single_pulse_search
#$ -cwd
#$ -S /bin/bash
#$ -t 1-1

. /etc/profile.d/modules.sh
module load gcc

export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot

dirs=(`ls -d */`)

echo "Starting Single Pulse Search: `date`"
python $PRESTO/bin/single_pulse_search.py -t 5 ${dirs[$SGE_TASK_ID-1]}/*.dat 
echo "Finished Single Pulse Search: `date`"
