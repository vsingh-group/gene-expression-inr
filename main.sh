# for ((i=0; i<58692; i+=20))
# do
#     for ((j=i; j<i+20 && j<58692; j++))
#     do
#         CUDA_VISIBLE_DEVICES=3 python train.py --index $j &
#     done
#     wait
# done

# CUDA_VISIBLE_DEVICES=2 python train_gene_net.py --config ./configs/config_sweep_lr.yaml
# CUDA_VISIBLE_DEVICES=7 python train_gene_net.py --config ./configs/config_sweep_features.yaml
# wait
CUDA_VISIBLE_DEVICES=0 python train_gene_net.py --config ./configs/config_sweep_hidden.yaml
# CUDA_VISIBLE_DEVICES=2 python train_gene_net.py --config ./configs/config_sweep_encode.yaml
# CUDA_VISIBLE_DEVICES=7 python train_gene_net_noisy.py --config ./configs/config_sweep_noisy.yaml