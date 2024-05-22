CUDA_VISIBLE_DEVICES=2 python train.py -s /NASdata/gj/datasets/nerf_synthetic/lego -m ./output_profile/torch_sort --profile --debug


CUDA_VISIBLE_DEVICES=7 python train.py -s /NASdata/gj/datasets/nerf_synthetic/lego -m ./output_profile/cuda_sort

CUDA_VISIBLE_DEVICES=1 python train.py -s /NASdata/gj/datasets/nerf_synthetic/lego -m ./output_profile/cuda_sort --eval --start_checkpoint ./output_profile/cuda_sort/chkpnt3530.pth