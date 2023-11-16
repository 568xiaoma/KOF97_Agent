#/bin/bash

# k=`xauth list | grep ${DISPLAY%??}`; xauth add $k
sudo env CUDA_VISIBLE_DEVICES=4 PATH=$PATH python king_of_fighter/debug.py --role kyo --player 1 --display $DISPLAY  