#!/bin/bash

# Amazon_Sports_and_Outdoors
python run_recbole.py --dataset='Amazon_Sports_and_Outdoors' --reg_weight=1e-2 --neg_sam_num=3 --learning_rate=0.001

# Amazon_Toys_and_Games
python run_recbole.py --dataset='Amazon_Toys_and_Games' --reg_weight=1e-9 --neg_sam_num=1 --learning_rate=0.005
#
# yelp
python run_recbole.py