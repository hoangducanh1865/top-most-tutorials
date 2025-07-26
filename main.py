import topmost 

from src.config.config import DEVICE
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import infer_model


dataset, model, trainer, top_words_en, top_words_cn, train_theta_en, train_theta_cn = train_model()
TD, results = evaluate_model(trainer, dataset, top_words_en, top_words_cn)
print(f"TD: {TD:.5f}")
print(results)

# infer_model(trainer, dataset)