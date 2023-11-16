#/bin/bash

# k=`xauth list | grep ${DISPLAY%??}`; xauth add $k
# CUDA_VISIBLE_DEVICES=7 \
sudo env PATH=$PATH CUDA_VISIBLE_DEVICES=7 python king_of_fighter/baseline.py --role kyo