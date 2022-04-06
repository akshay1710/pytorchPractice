from torch.utils.data import Dataset
from pathlib import Path
import json
import albumentations as A
import cv2
import torch
from pycocotools.coco import COCO
from PIL import Image
import os

class BCCDDataset(Dataset):
    def __init__(self, images_dir, label_json, h,w):
        self.images_dir =os.getcwd()
        self.annot =COCO(label_json)
        self.images_list =list(self.images_dir.iterdir())
        self.c2i = {'RBC': 0, 'WBC': 1, 'Platelets': 2}
        self.i2c = ['RBC', 'WBC', 'Platelets']
        self.h = h
        self.w = w
        self.ids = list(sorted(self.annot.imgs.keys()))
        self.augs = A.Compose([A.HorizontalFlip(p=0.5),
                              A.VerticalFlip(p=0.5),
                              A.Resize(h,w),
                              A.Normalize(mean=(0.485,0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
                              A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT)],
                              bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        img_name = self.images_list[item].name
        print(img_name)
        # img = cv2.imread(str(self.images_list[item]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ids = self.ids[item]
        ann_ids = COCO.getAnnIds(imgIds=img_ids)
        path = COCO.loadAnns(ann_ids)
        img = Image.open(os.path.join(self.root, path))
        num_objs = len(coco_annotation)

        bbox, labels_str =self.annot['images'][item]
        bbox = torch.tensor(bbox)
        labels =torch.tensor([self.c2i[i] for i in labels_str])
        augmented = self.augs(image=img, bboxes = bbox.tolist(), labels = labels.tolist())
        img = augmented['image']
        bbox = augmented['bboxes']
        labels = augmented['lables']
        bbox = torch.tensor(bbox)
        labels = torch.tensor(labels)
        return img, (bbox, labels)

img, out = BCCDDataset('./train', '_annotations.coco.json', 300,300)

