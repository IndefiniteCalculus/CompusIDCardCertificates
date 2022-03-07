import torch

from CharacterIdentification.model import ChineseCharNet as ccn_model
from CharacterIdentification import LabelReader
from CharacterIdentification.model import ChineseCharDataLoader
chars = LabelReader.get_labels()
labels = LabelReader.encoded_onehot_label()
model = ccn_model.ChineseCharNet()
model = model.load_state_dict(torch.load("chinese_char_model_100.pt"))

pass