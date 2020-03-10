#!/bin/bash

disassemble_gif() {
  convert $1 -coalesce $2-%04d.gif
}

concat_frames_h1() {
  for f in a-*.gif;
  do
    convert $f ${f/a/b} +append $f;
  done
}

concat_frames_h2() {
  for f in c-*.gif;
  do
    convert $f ${f/c/d} +append $f;
  done
}

concat_frames_v() {
  for f in a-*.gif;
  do
    convert $f ${f/a/c} -append $f;
  done
}

reassemble_gif() {
  convert -loop 1 -delay 20 a-*.gif $1
}

clean() {
  rm -f a-*.gif
  rm -f b-*.gif
  rm -f c-*.gif
  rm -f d-*.gif
}

concat4() {
  disassemble_gif $1 a
  disassemble_gif $2 b
  disassemble_gif $3 c
  disassemble_gif $4 d

  concat_frames_h1
  concat_frames_h2

  concat_frames_v

  reassemble_gif $5

  clean
}

if [[ $# != 5 ]];
then
  echo -e "\n\t$0 [-h|-v] input1.gif input2.gif input3.gif input4.gif output.gif\n"
  exit -1
fi

concat4 $1 $2 $3 $4 $5
