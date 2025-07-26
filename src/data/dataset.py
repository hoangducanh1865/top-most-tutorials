import topmost
from topmost import download_dataset
from src.config.config import DEVICE


def get_dataset():
    dataset_dir = "./datasets/20NG"
    download_dataset("20NG", cache_path="./datasets")
    dataset = topmost.data.BasicDataset(dataset_dir, read_labels=True, device=DEVICE, contextual_embed=False)
    print("train_size:", len(dataset.train_data) if hasattr(dataset, "train_data") else "N/A")
    print("test_size:", len(dataset.test_data) if hasattr(dataset, "test_data") else "N/A")
    print("vocab_size:", dataset.vocab_size if hasattr(dataset, "vocab_size") else "N/A")
    print("average length:", dataset.avg_len if hasattr(dataset, "avg_len") else "N/A")
    return dataset