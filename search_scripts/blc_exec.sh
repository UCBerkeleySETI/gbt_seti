#!/bin/bash
for h in blc{1..1}{0..7}
do
	echo $h
	ssh $h "$@"
done

