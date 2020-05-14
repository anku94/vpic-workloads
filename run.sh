#!/bin/bash

code=chi-square-test

python $code.py 'T.100/h*' &
python $code.py 'T.950/h*' &
python $code.py 'T.1900/h*'&
python $code.py 'T.2850/h*'&
