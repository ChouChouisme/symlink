#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: infer.sh

DATA_SIGN=semeval
FILE_NAME=mrc-ner.wxl
DATA_DIR=/data0/wxl/symlink/NER/data/input
NER_OUPUT_DIR=/data0/wxl/symlink/NER/data/output
RE_OUPUT_DIR=/data0/wxl/symlink/RE/result
BERT_DIR=/data0/wxl/symlink/NER
MAX_LEN=512
MODEL_CKPT=/data0/wxl/symlink/model/ner/epoch=304.ckpt
HPARAMS_FILE=/data0/wxl/symlink/model/ner/lightning_logs/version_1/hparams.yaml

python3 /data0/wxl/symlink/NER/inference.py \
--data_dir ${DATA_DIR}/${FILE_NAME} \
--output_dir ${NER_OUPUT_DIR}/${FILE_NAME} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}

python3 /data0/wxl/symlink/RE/inference.py \
--data_dir ${NER_OUPUT_DIR}/${FILE_NAME} \
--output_dir ${RE_OUPUT_DIR} \
--file_name ${FILE_NAME} \