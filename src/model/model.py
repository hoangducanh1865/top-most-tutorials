import topmost
from src.config.config import DEVICE


def get_model_and_trainer(dataset):
    '''Basic Model'''
    # model = topmost.ETM(dataset.vocab_size)
    # # model = topmost.DecTM(dataset.vocab_size)
    # # model = topmost.TSCTM(dataset.vocab_size)
    # # model = topmost.CombinedTM(dataset.vocab_size, dataset.contextual_embed_size)
    # # model = topmost.NSTM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    # # model = topmost.ECRTM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    # model = model.to(DEVICE)
    # trainer = topmost.BasicTrainer(model, dataset, verbose=True)
    
    # return model, trainer
    
    
    '''Crosslingual Model'''
    model = topmost.InfoCTM(
        trans_e2c=dataset.trans_matrix_en,
        pretrain_word_embeddings_en=dataset.pretrained_WE_en,
        pretrain_word_embeddings_cn=dataset.pretrained_WE_cn,
        vocab_size_en=dataset.vocab_size_en,
        vocab_size_cn=dataset.vocab_size_cn,
        weight_MI=50
    )
    model = model.to(DEVICE)
    trainer = topmost.CrosslingualTrainer(model, dataset, lr_scheduler='StepLR', lr_step_size=125, epochs=500)
    
    return model, trainer