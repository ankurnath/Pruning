from datasets import load_dataset
dataset = load_dataset("nateraw/food101")

# print(dir())

# from transformers import AutoFeatureExtractor, AutoModel
# from collections import defaultdict
# import pandas as pd
# model_ckpt = "Ahmed9275/Vit-Cifar100"
# # model_ckpt = 'google/vit-base-patch16-384'
# extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)
# hidden_dim = model.config.hidden_size
# # hidden_dim = model.config.hidden_size
# print(dataset['train'])

# print(model)