#!/bin/bash
experiment=${1:-"coco_train"}
experiment_name=${2:+"-n $2"}
model_name=${2:+"-expn $2"}
python -m yolox.tools.train -f ./exps/$experiment.py $experiment_name $model_name -b 16 --fp16 -o -c ./models/ycbv-gdrnpp_bop/model_final.pth