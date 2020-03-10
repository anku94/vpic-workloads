#!/bin/bash

disassemble_gif() {
  convert $1 -coalesce $2-%04d.gif
}

concat_frames_h() {
  for f in a-*.gif;
  do
    convert $f ${f/a/b} +append $f;
  done
}

concat_frames_v() {
  for f in a-*.gif;
  do
    convert $f ${f/a/b} -append $f;
  done
}

reassemble_gif() {
  convert -loop 1 -delay 20 a-*.gif $1
}

clean() {
  rm -f a-*.gif
  rm -f b-*.gif
}

if [[ $# != 4 ]];
then
  echo -e "\n\t$0 [-h|-v] input1.gif input2.gif output.gif\n"
  exit -1
fi

disassemble_gif $2 a
disassemble_gif $3 b

if [[ $1 == "-v" ]];
then
  echo "Running vertical concatenation... "

  concat_frames_v
else
  echo "Running horizontal concatenation... "

  concat_frames_h
fi

reassemble_gif $4

clean
