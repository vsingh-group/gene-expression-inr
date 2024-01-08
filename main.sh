for ((i=0; i<58692; i+=20))
do
    for ((j=i; j<i+20 && j<58692; j++))
    do
        CUDA_VISIBLE_DEVICES=3 python train.py --index $j &
    done
    wait
done
