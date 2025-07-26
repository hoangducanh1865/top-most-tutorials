from src.data.dataset import get_dataset
from src.model.model import get_model_and_trainer


'''Basic Model'''
# def train_model():
#     dataset = get_dataset()
#     model, trainer = get_model_and_trainer(dataset)
#     top_words, train_theta = trainer.train()
#     return dataset, model, trainer, top_words, train_theta


'''Crosslingual Model'''
def train_model():
    dataset = get_dataset()
    model, trainer = get_model_and_trainer(dataset)
    top_words_en, top_words_cn, train_theta_en, train_theta_cn = trainer.train()
    return dataset, model, trainer, top_words_en, top_words_cn, train_theta_en, train_theta_cn