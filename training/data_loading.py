# Using the MSCOCO dataset particularly the 2017 version
from pycocotools.coco import COCO
import requests
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
ann_file = "data/instances_train2017.json"
TEST_DIR = "data/coco_test"
TRAIN_DIR = "data/coco_train"
NUM_SAMPLES = 100
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)


# Initialize COCO API
coco = COCO(ann_file)

# Pick a category (e.g., "person")
cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)

url = []
for category in categories:
    cat_id = category['id']
    cat_name = category['name']

    img_ids = coco.getImgIds(catIds=[cat_id])
    sample_ids = random.sample(img_ids, NUM_SAMPLES)
    imgID = coco.loadImgs(sample_ids)
    for img in imgID:
        url.append(img['coco_url'])

print(f"Total samples: {len(url)}")
split_index = int(0.8 * len(url))
train_urls = url[:split_index]
test_urls = url[split_index:]

print(f"Length of train set: {len(train_urls)}")
print(f"Length of test set: {len(test_urls)}")


def download_and_save(idx, url, save_dir):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(os.path.join(save_dir, f"{idx}.jpg"))
    except Exception as e:
        print(f"Failed {url}: {e}")


def download_images(urls, save_dir, max_workers=None):
    os.makedirs(save_dir, exist_ok=True)

    if max_workers is None:
        cores = os.cpu_count()
        max_workers = min(200, cores * 10)
    print(f"Using {max_workers} workers for downloading images.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, url in enumerate(urls):
            executor.submit(download_and_save, i, url, save_dir)


download_images(train_urls, TRAIN_DIR)
download_images(test_urls, TEST_DIR)
