import topmost
from topmost import download_dataset
from src.config.config import DEVICE


def get_dataset():
    dataset_dir = "./datasets/20NG"
    download_dataset("20NG", cache_path="./datasets")
    dataset = topmost.data.BasicDataset(dataset_dir, read_labels=True, device=DEVICE, contextual_embed=True, doc_embed_model='all-MiniLM-L6-v2')
    return dataset