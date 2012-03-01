#$ -N SearchRun
#$ -cwd
#$ -S /bin/bash

# List all files to be processed
files=(`ls $HOME/SETI/KeplerData/*.fil`)
numFiles=$((${#files[*]}))

for (( i=0; i<$numFiles; i++))
do
    # Get current file name
    file=${files[$i]}
    filename=${file##*/}

    echo "Processing file $filename"
    ssh master "cd $HOME/SETI; qsub AlienSearch.sh $filename"
    sleep 7200
done

