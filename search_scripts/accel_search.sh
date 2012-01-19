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
export SETI=$HOME/SETI
export REMOTE_DIR=$HOME/SETI/Data/$1/
export SPLIT=0

# List all files to be processed
files=(`ssh master ls $HOME/SETI/Data/$1/*.dat`)
numFiles=$((${#files[*]} / 32))

echo $files

mkdir $1_$SGE_TASK_ID
cd $1_$SGE_TASK_ID

# Process all file for this job task
echo "Starting Search: `date`"

for (( i=0; i<$numFiles; i++))
do
    # Get current file name
    file=${files[$((($SGE_TASK_ID-1)*$numFiles+$i))]}
    filename=${file##*/}

    echo "Started $filename: `date`"

    # Copy files from remote to local server
    scp master:${file%\.dat}* .  

    # Perform single pulse search on file
    python $PRESTO/bin/single_pulse_search.py --noplot -t 5 $filename >> SearchOutput.txt

    # Copy single pulse results to remote store
    scp *singlepulse* master:$HOME/SETI/Data/$1

    # Only split files if required 
    if [ $SPLIT -eq 1 ] 
    then
        # Split files into 10 seconds chunks for periodicity search    
        python $SETI/splitData.py $filename ${filename%\.dat}_ chunk=10
 
        # Loop over all split chunks
        chunks=(`ls *.dat | grep _[0-9][0-9].dat`)
        numChunks=$((${#chunks[*]}))

        echo "Processing $filename: `date`"

        for (( j=0; j<$numChunks; j++))
        do
            $PRESTO/bin/accelsearch -zmax 0 -harmpolish ${chunks[$j]} >> SearchOutput.txt
        done
    else
        $PRESTO/bin/accelsearch -zmax 0 -harmpolish -flo 0.1 $filename >> SearchOutput.txt
    fi

    # Create tar file with results and copy to remote store
    tar -czf ${filename%\.dat}_ACCEL.tar.gz *ACCEL*
    scp *ACCEL.tar.gz master:$HOME/SETI/Data/$1

    # Remove files
    rm -f ${filename%\.dat}*

    # Remove file from remote store
#    ssh master rm $HOME/SETI/Data/$1/$filename

    echo "Finished $filename: `date`"
done

echo "Finished Search: `date`"

cd ..
rm -fr $1_$SGE_TASK_ID
