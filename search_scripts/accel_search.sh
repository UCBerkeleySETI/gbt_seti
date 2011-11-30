#$ -N accel_search
#$ -cwd
#$ -S /bin/bash
#$ -t 1-50

. /etc/profile.d/modules.sh
module load gcc

# Define environmental variables
export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot
export SETI=$HOME/SETI

# List all files to be processed
files=(`ssh master ls $HOME/SETI/Data/B1839+56_8bit.fil/*.dat`)
numFiles=$((${#files[*]} / 50))

mkdir $SGE_TASK_ID
cd $SGE_TASK_ID

# Process all file for this job task
echo "Starting Search: `date`"

for (( i=0; i<$numFiles; i++))
do
    # Get current file name
    file=${files[$((($SGE_TASK_ID-1)*$numFiles+$i))]}
    filename=${file##*/}

    # Copy dat and inf file for this file locally
    scp master:${file%\.dat}* .  

    # Perform single pulse search on file
    python $PRESTO/bin/single_pulse_search.py --noplot -t 5 $filename >> SearchOutput.txt

    # Copy single pulse results to remote store
    scp *singlepulse* master:$HOME/SETI/Data/B1839+56_8bit.fil/

    # Split files into 10 seconds chunks for periodicity search    
    python $SETI/splitData.py $filename ${filename%\.dat}_ chunk=10
 
    # Loop over all split chunks
    chunks=(`ls *.dat | grep _[0-9][0-9].dat`)
    numChunks=$((${#chunks[*]}))

    for (( j=0; j<$numChunks; j++))
    do
        $PRESTO/bin/accelsearch -zmax 0 -harmpolish ${chunks[$j]} >> SearchOutput.txt
    done

    # Copy cand results to remote store
    scp *ACCEL* master:$HOME/SETI/Data/B1839+56_8bit.fil/

    # Remove files
    rm -f ${filename%\.dat}*
done

echo "Finished Search: `date`"

cd ..
rm -fr $SGE_TASK_ID
