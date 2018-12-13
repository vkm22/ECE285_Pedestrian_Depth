#!/usr/bin/env bash

ROOT_DIR="../Pedestrian-Detection-modified"
protoc $ROOT_DIR/object_detection/protos/*.proto --python_out=$ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim

python $ROOT_DIR/object_detection/inference.py \
--input_dir=../input_images/left_view/ \
--output_dir=../RESULTS_faster_rcnn/left_view/ \
--label_map=$ROOT_DIR/annotations/label_map.pbtxt \
--frozen_graph=$ROOT_DIR/output/frozen_inference_graph.pb \
--num_output_classes=1 \
--n_jobs=1 \
--delay=0

