# /data/home/zouqihang/desktop/project/M2M/version0.4.2

accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py -d /fs/home/jiluzhang/scM2M_iConv/data -n test --batch_size 24 > logs/arg1_running_logs.txt
