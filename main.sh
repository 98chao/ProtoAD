# 'carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 
# 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper'
class=('carpet')
gpu_id='0'

for index in $(seq 0 `expr ${#class[*]} - 1`); do
{
    class_name=${class[${index}]}
    ckpt_dir="./prototype/"
    python main.py --class_name=$class_name --gpu_id=$gpu_id --ckpt_dir=$ckpt_dir
}
done
