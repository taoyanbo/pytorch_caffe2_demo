export MXNET_CPU_WORKER_NTHREADS=160
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train.py --gpus 0 \
    --data-train data/train.txt \
    --model-prefix 'models/simple-conv3' \
    --batch-size 80 --num-classes 2 --num-examples 900 2>&1 | tee log.txt

