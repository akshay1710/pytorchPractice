import json
from pycocotools.coco import COCO
import os
from PIL import Image
VAL = COCO('_annotations.coco.json')
val = json.load(open('_annotations.coco.json', 'r'))
OUT = list(sorted(VAL.imgs.keys()))
ann_ids = VAL.getAnnIds(imgIds=OUT[2:4])
coco_annotation = VAL.loadAnns(ann_ids)
print(coco_annotation)

path = VAL.loadImgs(OUT[2:4])[0]['file_name']
print(path)
root = os.getcwd()
root = os.path.join(root, 'train')
img_path = os.path.join(root,  path)
print(img_path)

img = Image.open(img_path)
print(img)
path =  os.getcwd()
print(path)