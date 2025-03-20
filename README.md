# DeDe -- CVPR 2025
## Repository of paper 'DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders' 

To run comparisons or initiate a victim encoder, you will need to download [DRUPE](https://github.com/Gwinhen/DRUPE), [DECREE](https://github.com/GiantSeaweed/DECREE), [CTRL](https://github.com/meet-cjli/CTRL), [BadCLIP](https://github.com/LiangSiyuan21/BadCLIP), [ASSET](https://github.com/reds-lab/ASSET) from their public repositories respectively. BTW, BadEncoder is implemented by DRUPE. 

You can refer to `./src/run.sh` for more code usage. A jupyter notebook file is included for a detailed pipeline and evaluation for DeDe along with other attack/defense comparisons. We present an example for training a DeDe model for BadEncoder attack in the following :  
> python3 -u main.py --attack_type badencoder --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type id  --gpu 0 --save_tag 'mask9_patch4_id'

If you want to use the out-of-distribution dataset (e.g. STL10) for training DeDe, change `--traindata_type`. Note: `--poison_rate` is used for in-distribution but poisoned training dataset. Hence, you can neglect this argument in the case of OoD dataset. 
> python3 -u main.py --attack_type badencoder --poison_rate 0.01 --mask_ratio 0.9 --patch_size 4 --traindata_type ood  --gpu 1 --save_tag 'mask9_patch4_ood'

The downstream performance is evaluated by running:
> python3 -u downstream_evaluation.py --attack_type badencoder --poison_rate 0.0 --epochs 100 --test_mask_ratio 0.99 --gpu 2

Here, `--poison_rate` is used to control the downstream training dataset. It is chosen as 0.0 for a clean training dataset. 

You can always choose to train your own victim encoder or obtain it from the above-mentioned repositories. DeDe is non-invasive in both phases of training and testing. You can try different parameters to get the best results. 
We provide a victim encoder trained by DRUPE(a stealthy backdoor attack) and a DeDe model in the following links: TODO (uploading).

If you find it useful, please cite:
TODO
