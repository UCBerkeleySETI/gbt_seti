mkdir /datax/scratch/nbsearch/`uname -n`
rm /datax/scratch/nbsearch/`uname -n`/*
cp /datax/scratch/nbsearch/*.fits /datax/scratch/nbsearch/`uname -n`
tar -cvf /datax/scratch/nbsearch/`uname -n`.tar /datax/scratch/nbsearch/`uname -n`
cp /datax/scratch/nbsearch/`uname -n`.tar /home/siemion/fits
