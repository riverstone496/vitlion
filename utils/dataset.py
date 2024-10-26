import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, requests
from PIL import Image
from torch.utils.data import Subset

def download_file(url, local_path):
    if not os.path.exists(local_path):  # ファイルが存在しない場合のみダウンロード
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            print(f"Error: File not found at {url}")
            return None  # ファイルが存在しない場合はNoneを返す
        response.raise_for_status()  # その他のHTTPエラーの場合に例外を投げる
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # フィルター処理されたチャンクが空ではないことを確認
                    f.write(chunk)
    return local_path

def load_cifar5m(local_dir='./data_cifar/', train=True):
    base_url = 'https://storage.googleapis.com/gresearch/cifar5m/'
    npart = 1000448
    
    print('Loading CIFAR 5mil...')
    if train:
        X_tr = np.empty((5*npart, 32, 32, 3), dtype=np.uint8)
        Ys = []
        for i in range(5):
            local_file = download_file(f'{base_url}part{i}.npz', os.path.join(local_dir, f'part{i}.npz'))
            if local_file is None or not os.path.exists(local_file):
                continue  # ファイルが存在しなかった場合やダウンロードが完了していない場合はスキップ
            try:
                z = np.load(local_file, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {local_file}: {e}")
                continue
            X_tr[i*npart: (i+1)*npart] = z['X']
            Ys.append(torch.tensor(z['Y']).long())
            print(f'Loaded part {i+1}/5')
        Y_tr = torch.cat(Ys) if Ys else None
        return X_tr, Y_tr
    else:
        local_file = download_file(f'{base_url}part5.npz', os.path.join(local_dir, 'part5.npz'))
        if local_file and os.path.exists(local_file):
            try:
                z = np.load(local_file, allow_pickle=True)
                print(f'Loaded part 6/6')
                nte = 10000
                X_te = z['X'][:nte]
                Y_te = torch.tensor(z['Y'][:nte]).long()
                return X_te, Y_te
            except Exception as e:
                print(f"Error loading {local_file}: {e}")

class CIFAR5mDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = [Image.fromarray(img) for img in X]  # すべての画像をPIL.Imageに変換
        self.Y = Y
        self.targets = Y.tolist()  
        self.transform = transform
        self.classes = list(range(10))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        img = self.X[idx]  # すでにPIL.Imageに変換された画像データ
        target = self.Y[idx]

        # transformが設定されていれば適用
        if self.transform:
            img = self.transform(img)

        return img, target

def get_class_subset(dataset, num_replicas, rank):
    indices = [i for i, (_, target) in enumerate(dataset) if target % num_replicas == rank]
    return Subset(dataset, indices)