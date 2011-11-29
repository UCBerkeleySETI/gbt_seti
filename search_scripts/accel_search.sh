#$ -N accel_search
#$ -cwd
#$ -S /bin/bash
#$ -t 1-32

. /etc/profile.d/modules.sh
module load gcc

# Define environmental variables
export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot

# List all files to be processed
files=(`ssh master ls $HOME/SETI/Data/B1839+56_8bit.fil/*.dat`)
numFiles=$((${#files[*]} / 32))

# Process all file for this job task
echo "Starting accel search: `date`"

for (( i=0; i<$numFiles; i++))
do
    # Get current file name
    file=${files[$((($SGE_TASK_ID-1)*$numFiles+$i))]}
    filename=${file##*/}

    # Copy dat and inf file for this file locally
    scp master:${file%\.dat}* .
  
    # Perform acceleration search on copied files
    $PRESTO/bin/accelsearch -zmax 0 -harmpolish $filename >> SearchOutput.txt

    # Copy cand results to remote store
    scp *ACCEL* master:$HOME/SETI/Data/B1839+56_8bit.fil/

    # Remove files
    rm -f ${filename%\.dat}*
done

echo "Finished accel search: `date`"

