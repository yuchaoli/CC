CUDA_VISIBLE_DEVICES=7 python3 -u compress.py --dataset cifar10 \
                                              --net densenet40 \
                                              --pretrained True \
                                              --checkpoint pth/densenet40.pth \
                                              --train_dir tmp/densenet40_CC_0.5 \
                                              --train_batch_size 128 \
                                              --com_ratio 0.5 > compress_1.log 2>&1 &