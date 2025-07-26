from topmost import eva
from src.metric.topic_diversity import multiaspect_diversity


'''Basic Model'''
def evaluate_model(trainer, dataset, top_words):
    train_theta, test_theta = trainer.export_theta()
    TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)
    TD = eva._diversity(top_words)    
    clt_results = eva._clustering(test_theta, dataset.test_labels)
    cls_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    return TC, TD, clt_results, cls_results


'''Crosslingual Model'''
def evaluate_model(trainer, dataset, top_words_en, top_words_cn):
    train_theta_en, train_theta_cn, test_theta_en, test_theta_cn = trainer.export_theta()
    
    # compute topic coherence (CNPMI)
    # refer to https://github.com/BobXWu/CNPMI
    TD = multiaspect_diversity((top_words_en, top_words_cn))

    results = eva.crosslingual_cls(
        train_theta_en,
        train_theta_cn,
        test_theta_en,
        test_theta_cn,
        dataset.train_labels_en,
        dataset.train_labels_cn,
        dataset.test_labels_en,
        dataset.test_labels_cn,
        classifier="SVM",
        gamma="auto"
    )
    return TD, results