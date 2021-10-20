# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from transformers import pipeline
import config

def main():

    nlp = pipeline('fill-mask', model=config.bert_model_dir, tokenizer=config.bert_model_dir)
    for i in nlp('[MASK]买的电脑要缴税吗'):
        print(i)

if __name__ == "__main__":
    main()
