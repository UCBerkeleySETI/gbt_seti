#$ -N accel_search
#$ -cwd
#$ -S /bin/bash
#$ -t 1-27

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
export ID=$SGE_TASK_ID

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

# Check if filename argument has been passed
if [ "$#" -eq 0 ]; then
    echo "Filename required as command-line argument"
    exit
fi

# Create working directory
if [ ! -d "ProcessedData/$1" ]; then
    mkdir ProcessedData/$1
fi
cd ProcessedData/$1

# Make sub-directory for easier file manipulation
if [ ! -d "$ID" ]; then
    mkdir $ID
fi
cd $ID

echo "Dedispersing $1"

# Subband dedisperse input file depending on SGE_TASK_ID
$PRESTO/bin/prepsubband -nobary -filterbank -dmstep ${dmStep[$ID-1]} -numdms $numDMs -lodm ${startDM[$ID-1]} -downsamp ${downSamp[$ID-1]} -mask ~/SETI/KeplerData/$1_rfifind.mask -o $1 ~/SETI/KeplerData/$1

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
    python $PRESTO/bin/single_pulse_search.py --noplot -t 5 $file >> SearchOutput.txt

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
            $PRESTO/bin/accelsearch -zmax 0 -harmpolish ${chunks[$j]} >> SearchOutput.txt
        done
    else
        $PRESTO/bin/accelsearch -zmax 0 -harmpolish -flo 0.1 $file >> SearchOutput.txt
    fi

    # Create tar file with results and copy to parent
    tar -czf ${file%\.dat}_ACCEL.tar.gz *ACCEL*
    cp *ACCEL.tar.gz ..
    cp *single* ..

    # Remove files
    rm -f ${file%\.dat}*

    echo "Finished $file: `date`"
done

echo "Finished Search: `date`"

cd ..
rm -fr $1
