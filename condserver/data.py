import torch
import io
import base64
import json
import torchvision
from torch.utils.data import DataLoader
from random import choice, randrange, seed, randint, uniform, shuffle
import numpy as np
import PIL
import math
from PIL import Image
import requests
from io import BytesIO
import webdataset
from webdataset.handlers import warn_and_continue
import concurrent.futures


CONNECTIONS = 16
TIMEOUT = 5
TARGET_SIZE = 256

URL_BATCH = 'http://127.0.0.1:4456/batch'
URL_CONDITIONING = 'http://127.0.0.1:4455/conditionings'

seed(3)


def b64_string_to_tensor(s: str) -> torch.Tensor:
    tens_bytes = base64.b64decode(s)
    buff = io.BytesIO(tens_bytes)
    buff.seek(0)
    return torch.load(buff, 'cpu')


def tensor_to_b64_string(tens: torch.Tensor) -> str:
    buff = io.BytesIO()
    torch.save(tens, buff)
    buff.seek(0)
    return base64.b64encode(buff.read()).decode("utf-8")


def resize_image(img):
    width, height = img.size   # Get dimensions

    rz_w = width
    rz_h = height
    _m = max(width, height)
    if _m == width:
        rz_w = math.floor((rz_w / rz_h) * TARGET_SIZE)
        rz_h = TARGET_SIZE
    if _m == height:
        rz_h = math.floor((rz_h / rz_w) * TARGET_SIZE)
        rz_w = TARGET_SIZE

    if rz_w < TARGET_SIZE:
        rz_w = TARGET_SIZE
    if rz_h < TARGET_SIZE:
        rz_h = TARGET_SIZE

    img = img.resize((rz_w, rz_h), resample=PIL.Image.LANCZOS)

    return img


def crop_random(img):
    img_size = img.size
    x_max = img_size[0] - TARGET_SIZE
    y_max = img_size[1] - TARGET_SIZE

    random_x = randrange(0, x_max//2 + 1) * 2
    random_y = randrange(0, y_max//2 + 1) * 2

    area = (random_x, random_y, random_x + TARGET_SIZE, random_y + TARGET_SIZE)
    c_img = img.crop(area)
    return c_img



class ProcessData:
    def __init__(self, image_size=TARGET_SIZE):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.RandomCrop(image_size),
        ])

    def __call__(self, data):
        data["jpg"] = self.transforms(data["jpg"])
        return data


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.convert('RGB')
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def collate_oldbookillustrations_2(batch):
    captions = []
    for i in batch:
        caption_1 = i.get('image_alt', None)
        caption_2 = i.get('image_caption', None)
        caption_3 = i.get('image_title', None)
        caption_4 = i.get('image_description', None)
        cs = choice(list(filter(lambda x: x is not None, [
            caption_1,
            caption_2,
            caption_3,
            caption_4,
        ])))
        subject = i.get('illustration_subject', None)
        rand_bool = uniform(0, 1) < 0.5
        rand_bool_20_percent = uniform(0, 1) < 0.2

        if subject is not None and rand_bool:
            cs = f'{subject}. {cs}'
        if subject is not None and not rand_bool:
            cs = f'{cs}. {subject}'

        tags = i.get('tags', None)
        if tags is not None:
            shuffle(tags)
            tags_s = ', '.join(tags)
            if rand_bool:
                cs = f'{cs}. {tags_s}'
            else:
                cs = f'{tags_s}. {cs}'
            if rand_bool_20_percent:
                cs = tags_s
        captions.append(cs)

    captions_flat_tensor = None
    captions_full_tensor = None
    uncaptions_flat_tensor = None
    uncaptions_full_tensor = None
    prior_flat = None
    prior_flat_uncond = None
    try:
        captions_json_resp = requests.post(URL_CONDITIONING,
            json={'captions': captions},
            timeout=600)
        assert captions_json_resp.status_code == 200
        captions_json = captions_json_resp.json()
        captions_flat_tensor = b64_string_to_tensor(captions_json['flat'])
        captions_full_tensor = b64_string_to_tensor(captions_json['full'])
        uncaptions_flat_tensor = b64_string_to_tensor(captions_json['flat_uncond'])
        uncaptions_full_tensor = b64_string_to_tensor(captions_json['full_uncond'])
        prior_flat = b64_string_to_tensor(captions_json['prior_flat'])
        prior_flat_uncond = b64_string_to_tensor(captions_json['prior_flat_uncond'])
    except Exception as e:
        print('failed to get caption tensors', e)
        import traceback
        traceback.print_exc()
        pass

    images = torch.cat([preprocess(crop_random(resize_image(i['1600px'])))
        for i in batch], 0)
    return images, {
        'captions': captions,
        'flat': captions_flat_tensor,
        'full': captions_full_tensor,
        'flat_uncond': uncaptions_flat_tensor,
        'full_uncond': uncaptions_full_tensor,
        'prior_flat': prior_flat,
        'prior_flat_uncond': prior_flat_uncond,
    }


def collate_laion_coco(
    batch,
    caption_key="TEXT",
    caption_keys=["top_caption", "all_captions"],
):
    images_pil = []
    failure_idxs = []

    def load_url(url_tup, timeout):
        # print('url tup', url_tup)
        url = url_tup[1]
        response = requests.get(url, timeout=timeout)
        img = Image.open(BytesIO(response.content))
        return (img, url_tup[0])

    # Just skip any URLs we fail to download.
    url_tups = [(idx, row['URL']) for idx, row in enumerate(batch)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        future_to_url = (executor.submit(load_url, url_t, TIMEOUT) for url_t in url_tups)
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                img_tup = future.result()
                if img_tup[0].size[0] >= TARGET_SIZE and \
                    img_tup[0].size[1] >= TARGET_SIZE:
                    images_pil.append(img_tup)
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                # print(f'Failed to get image at URL \'{urls[itr]}\'')
                pass

    images_pil = list(filter(lambda x: x is not None, images_pil))
    final_batch = []
    success_idxs = {val[1] for val in images_pil}
    failure_idxs = set(range(len(batch))) - success_idxs
    for idx, row in enumerate(batch):
        if idx in failure_idxs:
            continue
        if idx in success_idxs:
            img = next((x[0] for x in images_pil if x[1] == idx), None)
            if img is None:
                continue
            final_b_item = { 'img': img, **row }
            captions_choices = list(set(
                [final_b_item[caption_key]] +
                [final_b_item[caption_keys[0]]] +
                final_b_item[caption_keys[1]]
            ))
            final_b_item['chosen_caption'] = choice(captions_choices)
            final_batch.append(final_b_item)

    captions = [ i['chosen_caption'] for i in final_batch ]
    captions_flat_tensor = None
    captions_full_tensor = None
    uncaptions_flat_tensor = None
    uncaptions_full_tensor = None
    try:
        captions_json_resp = requests.post(URL_CONDITIONING,
            json={'captions': captions},
            timeout=60)
        assert captions_json_resp.status_code == 200
        captions_json = captions_json_resp.json()
        captions_flat_tensor = b64_string_to_tensor(captions_json['flat'])
        captions_full_tensor = b64_string_to_tensor(captions_json['full'])
        uncaptions_flat_tensor = b64_string_to_tensor(captions_json['flat_uncond'])
        uncaptions_full_tensor = b64_string_to_tensor(captions_json['full_uncond'])
    except Exception as e:
        print('failed to get caption tensors', e)
        import traceback
        traceback.print_exc()
        pass

    images = torch.cat([preprocess(crop_random(resize_image(i['img'])))
        for i in final_batch], 0)
    return images, {
        'captions': captions,
        'flat': captions_flat_tensor,
        'full': captions_full_tensor,
        'flat_uncond': uncaptions_flat_tensor,
        'full_uncond': uncaptions_full_tensor,
    }


class ProcessDataLaionCoco:
    def __init__(self):
        self.transforms = lambda img: preprocess(crop_random(resize_image(img)))

    def __call__(self, item,
        image_key="jpg",
        caption_key="txt",
        caption_keys=["top_caption", "all_captions"],
    ):
        output = {}

        image_data = item[image_key]

        output["image_filename"] = item["__key__"]
        image_data = item[image_key]
        image = Image.open(BytesIO(image_data))
        output["jpg"] = self.transforms(image)

        # list of txt + top_caption + all_captions
        captions = [item[caption_key]] + [item[caption_keys[1]]] + \
            item[caption_keys[2]]
        text = choice(captions)

        # Do we need this?? Why is text in bytes? Does all_captions need to be
        # decoded and json parsed first?
        caption = text.decode("utf-8")
        output["txt"] = caption # text 

        metadata_file = item["json"]
        metadata = metadata_file.decode("utf-8")
        output["metadata"] = metadata

        return output


class ProcessDataLaionA:
    def __init__(self):
        self.transforms = lambda img: preprocess(crop_random(resize_image(img)))

    def __call__(self, item,
        image_key="jpg",
        caption_key="txt",
    ):
        output = {}

        image_data = item[image_key]

        output["image_filename"] = item.get("__key__")
        image_data = item[image_key]
        image = Image.open(BytesIO(image_data))
        output["jpg"] = self.transforms(image)

        text = item[caption_key]
        caption = text.decode("utf-8")
        output["txt"] = caption

        metadata_file = item["json"]
        metadata = metadata_file.decode("utf-8")
        metadata = json.loads(metadata)
        output["metadata"] = metadata

        return [output]


def collate_laion_a(batch):
    batch = list(filter(filter_laion_a_dataset, batch))
    img_tensors = [i['jpg'] for i in batch]
    images = torch.cat(img_tensors, dim=0)
    return [images, [i['txt'] for i in batch]]


def filter_laion_coco_dataset(
    item,
    image_key="URL",
    caption_key="TEXT",
    caption_keys=["top_caption", "all_captions"],
    punsafe_key='punsafe',
    watermark_key='pwatermark',
    height_key='HEIGHT',
    width_key='WIDTH',
):
    if height_key not in item:
        return False
    if item[height_key] is not None and item[height_key] < TARGET_SIZE:
        return False
    if width_key not in item:
        return False
    if item[width_key] is not None and item[width_key] < TARGET_SIZE:
        return False
    if punsafe_key not in item:
        return False
    if item[punsafe_key] is not None and item[punsafe_key] > 0.99:
        return False
    if watermark_key not in item:
        return False
    if item[watermark_key] is not None and item[watermark_key] > 0.9:
        return False
    if caption_key not in item or item[caption_key] is None:
        return False
    if image_key not in item or item[image_key] is None:
        return False
    for c_k in caption_keys:
        if c_k not in item.keys() or item[c_k] is None:
            return False

    return True


def filter_laion_a_dataset(item,
    punsafe_key='punsafe',
    height_key='height',
    width_key='width',
):
    if "metadata" not in item:
        return False
    metadata = item['metadata']

    if height_key not in metadata:
        return False
    if metadata[height_key] < TARGET_SIZE:
        return False
    if width_key not in metadata:
        return False
    if metadata[width_key] < TARGET_SIZE:
        return False
    if punsafe_key not in metadata:
        return False
    if metadata[punsafe_key] > 0.99:
        return False

    return True


def get_dataloader(args):
    # for gigant/oldbookillustrations_2
    #
    import datasets
    dataset = datasets.load_dataset(args.dataset_path, split="train") \
        .shuffle(seed=randint(0, 2**32-1))
    dataloader = DataLoader(dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=16, # Work way ahead of time
        collate_fn=collate_oldbookillustrations_2)

    # For laion-coco
    # import datasets
    # dataset = datasets.load_dataset(args.dataset_path, split="train",
    #     streaming=True)
    # torch_iterable_dataset = dataset.with_format("torch")
    # dataloader = DataLoader(
    #     torch_iterable_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     collate_fn=collate_laion_coco)

    # for laion/laion-a
    # dataset = webdataset.WebDataset(
    #     args.dataset_path,
    #     resampled=True,
    #     handler=warn_and_continue,
    # ) \
    #     .map(
    #         ProcessDataLaionA(),
    #         handler=warn_and_continue,
    #     )

    # dataloader = DataLoader(
    #     dataset.batched(args.batch_size),
    #     batch_size=None,
    #     num_workers=args.num_workers,
    #     collate_fn=collate_laion_a,
    # )

    return dataloader


def get_dataloader_laion_coco(args):
    import datasets

    dataset = datasets \
        .load_dataset(args.dataset_path, split="train", streaming=True, cache_dir=args.cache_dir) \
        .shuffle(seed=randint(0, 2**32-1)) \
        .filter(filter_laion_coco_dataset)

    # dataset = datasets \
    #     .load_dataset(args.dataset_path, split="train", cache_dir='/scratch/cache')
    # num_shards = 255
    # shards = [ds.shard(num_shards=num_shards, index=index, contiguous=True) for index in range(num_shards)]

    def gen_from_shards(shards):
        for shard in shards:
            for example in shard:
                yield example

    # dataset_it = datasets.IterableDataset.from_generator(gen_from_shards, gen_kwargs={"shards": shards[:num_shards]}) \
    #     .filter(filter_laion_coco_dataset)

    def _collate(batch):
        return collate_laion_coco(batch)

    torch_iterable_dataset = dataset.with_format("torch")
    dataloader = DataLoader(
        torch_iterable_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=16, # Work way ahead of time
        collate_fn=_collate)
    print('Dataloader initialized')

    return dataloader
