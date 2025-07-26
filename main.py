import topmost 

from src.config.config import DEVICE
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import infer_model


dataset, model, trainer, top_words, train_theta = train_model()
TC, TD, clt_results, cls_results = evaluate_model(trainer, dataset, top_words)
print(f"TD: {TD:.5f}")
print(clt_results)
print(cls_results)

infer_model()