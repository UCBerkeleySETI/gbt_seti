#!/bin/bash
source /usr/local/listen/listen.sh
cd /datax/scratch/nbsearch
/usr/local/listen/bin/filterbanksearch -a ./B00020000.fil -b ./B00030000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00030000.fil -b ./B00020000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00040000.fil -b ./B00050000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00050000.fil -b ./B00040000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00060000.fil -b ./B00070000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00070000.fil -b ./B00060000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00080000.fil -b ./B00090000.fil
/usr/local/listen/bin/filterbanksearch -a ./B00090000.fil -b ./B00080000.fil

