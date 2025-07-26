import torch
from topmost import Preprocess
from src.config.config import DEVICE


def infer_model(trainer, dataset):
    preprocess = Preprocess(
        
    )
    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]

    _, new_bow = preprocess.parse(new_docs, vocab=dataset.vocab)
    new_theta = trainer.test(torch.as_tensor(new_bow, device=DEVICE).float())

    print(new_theta.argmax(1))