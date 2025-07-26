import topmost
from config.config import DEVICE


def get_model_and_trainer(dataset):
    model = topmost.ETM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    # model = topmost.DecTM(dataset.vocab_size)
    # model = topmost.TSCTM(dataset.vocab_size)
    # model = topmost.CombinedTM(dataset.vocab_size, dataset.contextual_embed_size)
    # model = topmost.NSTM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    # model = topmost.ECRTM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    model = model.to(DEVICE)
    trainer = topmost.BasicTrainer(model, dataset, verbose=True)
    
    return model, trainer