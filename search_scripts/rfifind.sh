#$ -N rfifind
#$ -cwd
#$ -S /bin/bash

. /etc/profile.d/modules.sh
module load gcc

# Define environmental variables
export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot
export SETI=$HOME/SETI

cd ~/SETI/KeplerData

# Use Presto rfifind 
$PRESTO/bin/rfifind -time 1.5 -o $1 $1
