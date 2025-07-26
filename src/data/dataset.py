from topmost import download_dataset


def get_dataset():
    dataset_dir = "./datasets/20NG"
    download_dataset("20NG", cache_path="./datasets")