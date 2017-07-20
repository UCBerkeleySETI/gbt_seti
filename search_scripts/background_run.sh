#!/bin/bash
nohup $@ > ~siemion/html/`uname -n`.log 2>&1 &
