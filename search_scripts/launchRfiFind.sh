# Define environmental variables
export SETI=$HOME/SETI

cd $SETI/KeplerData

# List all files to be processed
files=(`ls *.fil`)
numFiles=$((${#files[*]}))

# Loop over all the files
for (( i=0; i<$numFiles; i++))
do
  if [ ! -f ${files[$i]}_rfifind.mask ]
  then
    # Use Presto rfifind 
    qsub $SETI/rfifind.sh ${files[$i]}
  fi
done
