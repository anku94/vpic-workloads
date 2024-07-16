#!/bin/bash

idx=0
for i in `seq 200 3200 19400`; do
  echo $i, $idx
  mkdir T.$i
  mv hist.data.$idx T.$i/hist
  idx=$(( idx + 1 ))
done
