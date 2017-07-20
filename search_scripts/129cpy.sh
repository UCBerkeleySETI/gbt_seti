#!/bin/bash
shopt -s extglob
IFS='blc' read -a myarray <<< `uname -n`
NODENUM=${myarray[0]##*( )}
mkdir /home/siemion/plots/${NODENUM}
cp /datax/scratch/nbsearch/boyajian/*eps /home/siemion/plots/${NODENUM}

