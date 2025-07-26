from topmost import eva


def evaluate_model(trainer, dataset, top_words):
    train_theta, test_theta = trainer.export_theta()
    TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)
    TD = eva._diversity(top_words)    
    clt_results = eva._clustering(test_theta, dataset.test_labels)
    cls_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    return TC, TD, clt_results, cls_results

    
    