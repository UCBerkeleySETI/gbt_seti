#!/bin/bash
# ------------------------------------------
# Copyright 2017 Pragaash Ponnusamy
# Observation Tracker PubSub Console
# ------------------------------------------

START_PATTERN_ID="--id=([a-zA-Z0-9._]+)"
START_PATTERN_NODE_COUNT="--node-count=([1-2][0-9]|3[0-2]|[1-9])"
RUN_PATTERN_NODE="--node=(blc[0-3][0-7])"
RUN_PATTERN_STATUS="--status=([0-2])"
TOPIC="obs-tracker"
PUBSUB="gcloud beta pubsub topics publish $TOPIC"

case $1 in
  s|start)
  if [[ $2 =~ $START_PATTERN_ID  ]]; then
    ID=${BASH_REMATCH[1]}
    if [[ $3 =~ $START_PATTERN_NODE_COUNT ]]; then
      COUNT=${BASH_REMATCH[1]}
      eval "$PUBSUB \"OBS_START\" --attribute=\"id=$ID,nodeCount=$COUNT\""
    else
        echo ERROR: Node Count Required \& Should Be Valid
        exit 1
    fi
  else
    echo ERROR: Observation ID Required \& Should Be Valid
    exit 1
  fi
  ;;
  r|run)
  if [[ $2 =~ $RUN_PATTERN_NODE ]]; then
    NODE=${BASH_REMATCH[1]}
    if [[ $3 =~ $RUN_PATTERN_STATUS ]]; then
      STATUS=${BASH_REMATCH[1]}
      if [[ $4 =~ $START_PATTERN_ID ]]; then
        ID=${BASH_REMATCH[1]}
        eval "$PUBSUB \"OBS_RUN\" --attribute=\"observationId=$ID,node=$NODE,status=$STATUS\""
      else
        echo ERROR: Observation ID Required \& Should Be Valid
        exit 1
      fi
    else
      echo ERROR: Node Status Required \& Should Be Valid
      exit 1
    fi
  else
    echo ERROR: Node Name Required \& Should Be Valid
    exit 1
  fi
  ;;
  *)
  echo ERROR: Unrecognized option "($1)"
  exit 1
  ;;
esac
