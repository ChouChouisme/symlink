#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: infer.sh

DATA_SIGN=semeval
DATA_DIR=/data0/wxl/symlink/NER/data/input/mrc-ner.wxl
NER_OUPUT_DIR=/data0/wxl/symlink/NER/data/output/mrc-ner.wxl
RE_OUPUT_DIR=/data0/wxl/symlink/RE/result/mrc-ner.wxl
BERT_DIR=/data0/wxl/symlink/NER
MAX_LEN=512
MODEL_CKPT=/data0/wxl/symlink/model/ner/epoch=304.ckpt
HPARAMS_FILE=/data0/wxl/symlink/model/ner/lightning_logs/version_1/hparams.yaml

python3 /data0/wxl/symlink/NER/inference.py \
--data_dir ${DATA_DIR} \
--output_dir ${NER_OUPUT_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}

python3 /data0/wxl/symlink/RE/inference.py \
--data_dir ${NER_OUPUT_DIR} \
--output_dir ${RE_OUPUT_DIR} \