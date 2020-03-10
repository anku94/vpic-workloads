#!/bin/bash

code=chi-square-test

python $code.py 'T.100/e*' &
python $code.py 'T.950/e*' &
python $code.py 'T.1900/e*'&
python $code.py 'T.2850/e*'&
