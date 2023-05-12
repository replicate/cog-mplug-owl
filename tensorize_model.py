#!/usr/bin/env python
import torch
import os
import argparse
import logging 
import sys

from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoConfig

from interface import get_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

MODEL_PATH = "model/instruction_tuned.pth"
TOKENIZER_PATH = "model/tokenizer.model"
TENSORIZED_WEIGHTS_PATH = "model/mplug-owl.tensors"

print('Loading model...')
model, tokenizer, img_processor = get_model(
    checkpoint_path=MODEL_PATH, 
    tokenizer_path="model/tokenizer.model"
)

model.to('cuda')

print(f'Writing tensorized weights to {TENSORIZED_WEIGHTS_PATH}...')

serializer = TensorSerializer(TENSORIZED_WEIGHTS_PATH)
serializer.write_module(model)
serializer.close()
