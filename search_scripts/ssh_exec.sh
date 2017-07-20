#!/bin/bash
for h in blc{0..2}{0..7}
do
	ssh $h "nohup $@ > ~siemion/html/`uname -n`.log 2>&1 &"
done
