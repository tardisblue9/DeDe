#!/bin/bash


echo 'Calling scripts!'

rm -rf logs # toggle this off if you want to keep old logs each time you run new experiments

# Trian DEDE decoder model
python3 -u main.py --attack_type badencoder --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type id  --gpu 0 --save_tag 'mask9_patch4_id' &
python3 -u main.py --attack_type badencoder --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type ood  --gpu 1 --save_tag 'mask9_patch4_ood' &

python3 -u main.py --attack_type drupe --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type id  --gpu 2 --save_tag 'mask9_patch4_id' &
python3 -u main.py --attack_type drupe --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type ood  --gpu 3 --save_tag 'mask9_patch4_ood'

python3 -u main.py --attack_type ctrl --poison_rate 0.01 --mask_ratio 0.75 --patch_size 4 --traindata_type id --gpu 1 --save_tag 'mask75_patch4_id' &
python3 -u main.py --attack_type ctrl --poison_rate 0.01 --mask_ratio 0.75 --patch_size 4 --traindata_type ood --gpu 2 --save_tag 'mask75_patch4_ood'

python3 -u main.py --attack_type clip --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type id --epochs 100 --batch_size 16 --gpu 1 --save_tag 'mask9_patch4_id' &
python3 -u main.py --attack_type clip --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type ood --epochs 100 --batch_size 16 --gpu 2 --save_tag 'mask9_patch4_ood'

python3 -u main.py --attack_type badclip --poison_rate 0.01 --mask_ratio 0.9 --patch_size 32 --traindata_type id --epochs 100 --batch_size 16 --gpu 1 --save_tag 'mask9_patch32_id' &
python3 -u main.py --attack_type badclip --poison_rate 0.01 --mask_ratio 0.9 --patch_size 32 --traindata_type ood --epochs 100 --batch_size 16 --gpu 2 --save_tag 'mask9_patch32_ood'

# ASSET eval
python3 -u Asset_evaluation.py --attack_type badencoder  --poison_rate 0.01 --test_poison_rate 0.5 --epochs 10 --gpu 1
#python3 -u Asset_evaluation.py --attack_type badencoder  --poison_rate 0.2 --test_poison_rate 0.5 --epochs 10 --gpu 1
#python3 -u Asset_evaluation.py --attack_type drupe   --poison_rate 0.01 --test_poison_rate 0.5 --epochs 10 --gpu 2
#python3 -u Asset_evaluation.py --attack_type drupe   --poison_rate 0.2 --test_poison_rate 0.5 --epochs 10 --gpu 2
python3 -u Asset_evaluation.py --attack_type ctrl  --poison_rate 0.01 --test_poison_rate 0.5 --epochs 30 --gpu 3
#python3 -u Asset_evaluation.py --attack_type ctrl  --poison_rate 0.2 --test_poison_rate 0.5 --epochs 10 --gpu 3
#python3 -u Asset_evaluation.py --attack_type clip  --poison_rate 0.01 --test_poison_rate 0.5 --epochs 10 --gpu 1 &
python3 -u Asset_evaluation.py --attack_type clip  --poison_rate 0.2 --test_poison_rate 0.5 --epochs 10 --gpu 2
python3 -u Asset_evaluation.py --attack_type badclip  --poison_rate 0.01 --test_poison_rate 0.5 --epochs 10 --gpu 3
python3 -u Asset_evaluation.py --attack_type badclip  --poison_rate 0.2 --test_poison_rate 0.5 --epochs 10 --gpu 3

# DEDE downstream eval
#python3 -u downstream_evaluation.py --attack_type badencoder --poison_rate 0.0 --epochs 100 --test_mask_ratio 0.99 --gpu 3
#python3 -u downstream_evaluation.py --attack_type badencoder --poison_rate 0.01 --epochs 100 --test_mask_ratio 0.99 --gpu 3
#python3 -u downstream_evaluation.py --attack_type drupe --poison_rate 0.01 --epochs 100 --test_mask_ratio 0.99 --gpu 3
#python3 -u downstream_evaluation.py --attack_type ctrl --poison_rate 0.01 --epochs 100 --test_mask_ratio 0.99 --gpu 3
#python3 -u downstream_evaluation.py --attack_type clip --poison_rate 0.01 --epochs 10 --test_mask_ratio 0.99 --gpu 3
#python3 -u downstream_evaluation.py --attack_type badclip --poison_rate 0.01 --epochs 10 --test_mask_ratio 0.99 --gpu 3


echo 'All experiments are finished!'