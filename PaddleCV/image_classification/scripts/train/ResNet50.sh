##Training details
#GPU: NVIDIA® Tesla® V100 4cards 120epochs 67h
export CUDA_VISIBLE_DEVICES=4
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

export FLAGS_sync_nccl_allreduce=1
#export FLAGS_cudnn_exhaustive_search=1

#export GLOG_v=1
#export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO
# Unset proxy
unset https_proxy http_proxy

#MB
#export FLAGS_conv_workspace_size_limit=7000
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNet50:
python train.py \
       --model=ResNet50 \
       --batch_size=128 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=120 \
       --data_dir=./ImageNet \
       --use_fp16=True \
       --lr=0.1 \
       --l2_decay=1e-4
