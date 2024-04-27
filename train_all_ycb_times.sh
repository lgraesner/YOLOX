#!/bin/bash
runs=${1:-5}
output_dir=${2}
for i in $(seq $runs); do
    echo "### RUN $i ###"
    ./start_training.sh ycb_video_mixed "ycb_video_mixed_$i"
    ./start_training.sh ycb_video_external "ycb_video_external_$i"
    ./start_training.sh ycb_video_internal "ycb_video_internal_$i"
    if [ -z "$output_dir" ]
        mv ./YOLOX_outputs/ycb_video_mixed_$i $output_dir/ycb_video_mixed/$i &
        mv ./YOLOX_outputs/ycb_video_external$i $output_dir/ycb_video_external/$i &
        mv ./YOLOX_outputs/ycb_video_internal$i $output_dir/ycb_video_internal/$i &

    fi
done
echo "### DONE ###"