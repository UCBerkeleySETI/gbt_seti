#!/bin/bash
for h in blc{0..3}{0..7}
do
	echo $h
	ssh root@$h "$@"
done

