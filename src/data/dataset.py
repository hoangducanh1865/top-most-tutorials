import topmost
from topmost import download_dataset
from src.config.config import DEVICE


def get_dataset():
    '''Basic Model'''
    # dataset_dir = "./datasets/20NG"
    # download_dataset("20NG", cache_path="./datasets")
    # dataset = topmost.data.BasicDataset(dataset_dir, read_labels=True, device=DEVICE, contextual_embed=False)
    # print("train_size:", len(dataset.train_data) if hasattr(dataset, "train_data") else "N/A")
    # print("test_size:", len(dataset.test_data) if hasattr(dataset, "test_data") else "N/A")
    # print("vocab_size:", dataset.vocab_size if hasattr(dataset, "vocab_size") else "N/A")
    # print("average length:", dataset.avg_len if hasattr(dataset, "avg_len") else "N/A")
    # return dataset
    
    
    '''Crosslingual Model'''
    dataset_dir = "./datasets/Amazon_Review"
    download_dataset("Amazon_Review", cache_path="./datasets")
    dict_dir = "./datasets/dict"
    download_dataset("dict", cache_path="./datasets")
    dataset = topmost.CrosslingualDataset(dataset_dir, lang1="en", lang2="cn", dict_path=f"{dict_dir}/ch_en_dict.dat", device=DEVICE, batch_size=128) # QUESTION: Why 128?
    return dataset