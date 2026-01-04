import torch.utils.data as data
from PIL import Image
import os


class MyImageFolder(data.Dataset):
    def __init__(self, dir, transforms=None):
        image_list = []
        for target in sorted(os.listdir(dir)):
            path = os.path.join(dir, target)
            image_list.append(path)

        self.image_list = image_list
        self.transform = transforms

    def __getitem__(self, index):
        path = self.image_list[index]
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_list)
