#!/bin/bash
WEBDIR=~siemion/html
THISMACHINE=`uname -n`
INPUTDIRECTORY=`dirname $1`
INPUTFILE=`basename $1`
source /usr/local/pulsar64/pulsar.bash
source /usr/local/sigproc/bin/sigproc.sh
cd $INPUTDIRECTORY
echo $WEBDIR/${THISMACHINE}_
/usr/local/listen/bin/gpuspec -i $INPUTFILE -b 8 -B 2 -f 8 -t 262144 -o $WEBDIR/${THISMACHINE}_
cd $WEBDIR
reader ${THISMACHINE}_0000.fil > ${THISMACHINE}.txt