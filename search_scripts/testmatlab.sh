#!/bin/bash
find /datax/scratch/nbsearch -name \*fits -exec /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -r "fits_plot('{}',0.1,0.9,10,'{}','dark',0); quit" \;
