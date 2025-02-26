#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/data0/wxl/symlink/NER
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=semeval
DATA_DIR=/data0/wxl/symlink/NER/data/semeval
BERT_DIR=/data0/wxl/symlink/NER
MAX_LEN=512
MODEL_CKPT=/data0/wxl/symlink/output/ner/epoch=304.ckpt
HPARAMS_FILE=/data0/wxl/symlink/output/ner/lightning_logs/version_1/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}