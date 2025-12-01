import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CSIRO_Dataset(Dataset):
    def __init__(self, cfg, image_list):
        super().__init__()
        self.train_csv_path = cfg.train_csv_path
        self.data_root = cfg.data_root
        self.target_order = cfg.target_order   # e.g. ["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"]
        self.image_list = image_list
        assert isinstance(image_list, list)
        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_list)

    def _load_image(self, rel_path):
        path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        image_rel = self.image_list[index]

        # load image
        img = self._load_image(image_rel)

        # 根据 target_order 组装 5 维 vector
        tdict = self.groups[image_rel]
        targets = torch.tensor(
            [tdict[name] for name in self.target_order],
            dtype=torch.float32
        )
        targets = torch.log1p(targets)

        return {
            "image": img,
            "targets": targets,
            "image_path": image_rel
        }