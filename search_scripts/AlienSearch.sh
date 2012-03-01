#! /bin/sh
#$ -N alienSearch
#$ -cwd
#$ -S /bin/bash
#$ -t 1-27

# Create lockfile for mutual exclusion of multiple instances on one node
setup_lockfile()
{
    # set name of this program's lockfile:
    LOCKFILE=/local/amag0001/lock.$FILENAME.$$
    echo "Lockfile name is ${LOCKFILE}"
    sleep 1
}

# Check if lockfile exists
lock()
{
    # Check for an existing lock file
    while [ -f /local/amag0001/lock.$FILENAME* ]; do
        sleep 60
    done

    touch ${LOCKFILE}
    # check we managed to make it ok...
    if [ ! -f ${LOCKFILE} ]; then
        echo "Unable to create lockfile ${LOCKFILE}!"
        exit 1
    fi
    echo "Created lockfile ${LOCKFILE}"
}

# Cleanup lockfile
unlock()
{
    rm -f ${LOCKFILE}
    if [ -f ${LOCKFILE} ]; then
        echo "Unable to delete lockfile ${LOCKFILE}!"
        exit 1
    fi
    echo "Lock file ${LOCKFILE} removed."
}

. /etc/profile.d/modules.sh
module load gcc

# Define environmental variables
export FILENAME=$1
export PATH=$PATH:/usr/local/bin:/home/amag0001/tempo2
export PGPLOT_DIR=/opt/pgplot
export PRESTO=/home/amag0001/presto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib:$PRESTO/lib64:/opt/pgplot
export SETI=$HOME/SETI
export SPLIT=1
export ID=$SGE_TASK_ID
export DEL_FILE=0

# Define dedispersion parameters for each thread
startDM=( 0.00 10.24 20.48 30.72 40.96 51.20 61.44 71.68 81.92 92.16 102.40 
          112.64   138.24   163.84 
          189.44   240.64   291.84 
          343.04   496.64   650.24 
          803.84   1059.84  1315.84  
          1571.84  2083.84  2595.84
          3107.84  )

dmStep=(  0.02 0.02  0.02  0.02  0.02  0.02  0.02  0.02  0.02  0.02  0.02   
          0.05   0.05   0.05   
          0.1    0.1    0.1    
          0.3    0.3    0.3    
          0.5    0.5    0.5      
          1      1      1
          2 )

downSamp=( 1 1 1 1 1 1 1 1 1 1 1
           2  2  2
           4  4  4
           8  8  8
           16 16 16 
           32 32 32 
           64 )

export numDMs=512 # All instances process the same amount of threads

# Initialise lockfile
setup_lockfile

# Check if filename argument has been passed
if [ "$#" -eq 0 ]; then
    echo "Filename required as command-line argument"
    exit
fi

# Wait for a few seconds to prevent processes on the same node from running at the same time
sleep $ID

# Create working directories
if [ ! -d "/local/amag0001" ]; then
    mkdir /local/amag0001
fi

lock
if [ ! -d "/local/amag0001/$1_file" ] 
then  # First process on node, copy input file
    mkdir /local/amag0001/$1_file 

    # Copy data to local store
    echo "Copying data to local store"
    cp ~/SETI/KeplerData/$1 /local/amag0001/$1_file

    # Create process counter in counter file and set process as deletion process
    echo "1" > /local/amag0001/$1_file/$1.done
    export DEL_FILE=1

else  # Not first process on node, wait for file to be copied
    # Increment process counter for input file (ensure that only one instance executes this)
    export VALUE=`cat /local/amag0001/$1_file/$1.done`
    echo `expr $VALUE + 1` > /local/amag0001/$1_file/$1.done
    echo "Incremented counter from $VALUE to `expr $VALUE + 1`"
fi
unlock

if [ ! -d "/local/amag0001/$1_file/$ID" ]; then
    mkdir /local/amag0001/$1_file/$ID
fi

export WORK_DIR=/local/amag0001/$1_file/$ID
cd $WORK_DIR

if [ ! -d "$SETI/ProcessedData/$1" ]; then
    mkdir $SETI/ProcessedData/$1
fi

export OUTPUT_DIR=$SETI/ProcessedData/$1

# Make sub-directory for easier file manipulation
echo "Dedispersing $1"

# Subband dedisperse input file depending on SGE_TASK_ID
$PRESTO/bin/prepsubband -nobary -filterbank -dmstep ${dmStep[$ID-1]} -numdms $numDMs -lodm ${startDM[$ID-1]} -downsamp ${downSamp[$ID-1]} -mask ~/SETI/KeplerData/$1_rfifind.mask -o $1 ../$1

# Finished processing input file, decrease counter (use mutual exclusion)
lock
export VALUE=`cat /local/amag0001/$1_file/$1.done`
echo `expr $VALUE - 1` > /local/amag0001/$1_file/$1.done
unlock

# Process all files for this job task
echo "Starting Search: `date`"

# Get list of files to be processed by this process
files=(`ls *.dat`)

for (( i=0; i<$numDMs; i++))
do
    # Get current file name
    file=${files[$i]}
    
    echo "Started $file: `date`"

    # Perform single pulse search on file
    python $PRESTO/bin/single_pulse_search.py --noplot -t 5 $file > SearchOutput.txt

    # Only split files if required 
    if [ $SPLIT -eq 1 ] 
    then
        # Split files into 10 seconds chunks for periodicity search    
        python $SETI/splitData.py $file ${file%\.dat}_ chunk=10
 
        # Loop over all split chunks
        chunks=(`ls *.dat | grep _[0-9][0-9].dat`)
        numChunks=$((${#chunks[*]}))

        echo "Processing $file: `date`"

        for (( j=0; j<$numChunks; j++))
        do
            $PRESTO/bin/accelsearch -zmax 0 -harmpolish ${chunks[$j]} > SearchOutput.txt
        done
    else
        $PRESTO/bin/accelsearch -zmax 0 -harmpolish -flo 0.1 $file > SearchOutput.txt
    fi

    # Create tar file with results and copy to parent
    tar -czf ${file%\.dat}.tar *ACCEL* *single* *inf
    mv ${file%\.dat}.tar $OUTPUT_DIR

    # Remove files
    rm -f ${file%\.dat}*

    echo "Finished $file: `date`"
done

echo "Finished Search: `date`"

# Wait for all other processes before deleting file, if process is designated deleter
if [ $DEL_FILE -eq 0 ]
then
    while [ `cat /local/amag0001/$1_file/$1.done` -ne 0 ]; do
        sleep 60
    done

    rm /local/amag0001/$1_file/$1.done
    rm /local/amag0001/$1_file/$1
fi

# Remove working directory
rm -fr $WORK_DIR
