import pandas as pd 

import torch

import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from tqdm import tqdm

from multiprocessing import Pool

import random
import augly.image as imaugs

import itertools



NUM_WORKERS = 8

DATA_PATH = '../data/raw/'
OUT_PATH = '../data/processed/'

DTYPE = torch.uint8

loader = T.Compose([
    T.ToTensor()
]) 
NN_TRANSFORMS = torch.nn.Sequential(
    T.Resize((256,256), interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    T.ConvertImageDtype(DTYPE)
)
NN_TRANSFORMS = torch.jit.script(NN_TRANSFORMS)

def transform(args):
        
        id_, augment = args
        # Get anchor
        anchor = Image.open(DATA_PATH+'images/'+id_+'.jpg')
        query = NN_TRANSFORMS(loader(anchor.convert('RGB')))
        torch.save(query,OUT_PATH+'images/'+id_+'.pt')

        if not augment:
            return

        # Get Positive
        COLOR_JITTER_PARAMS = {
                "brightness_factor": random.uniform(0.3, 1.7),
                "contrast_factor": random.uniform(0.3, 1.7),
                "saturation_factor": random.uniform(0.3, 1.7),
        }

        AUGMENTATIONS = [
            imaugs.OneOf(
                [imaugs.Blur(),imaugs.Pixelization()]
            ),
            imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
            imaugs.OneOf(
                [imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText()]
            ),
        ]

        AUG = imaugs.Compose(AUGMENTATIONS)
        positive = NN_TRANSFORMS(loader(AUG(anchor).convert('RGB')))
        torch.save(positive,OUT_PATH+'augmented/'+id_+'.pt')


if __name__ == '__main__':

    reference = pd.read_csv(DATA_PATH + 'reference_images_metadata.csv').iloc[:1000,:]
    query = pd.read_csv(DATA_PATH + 'query_images_metadata.csv').iloc[:1000,:]
    training = pd.read_csv(DATA_PATH + 'training_images_metadata.csv').iloc[:1000,:]

    pool = Pool(processes=NUM_WORKERS)

    for _ in tqdm(pool.imap_unordered(transform, zip(training.image_id,itertools.repeat(True))), total=len(training.image_id)):
        pass

    for _ in tqdm(pool.imap_unordered(transform, zip(reference.image_id,itertools.repeat(False))), total=len(reference.image_id)):
        pass

    for _ in tqdm(pool.imap_unordered(transform, zip(query.image_id,itertools.repeat(False))), total=len(query.image_id)):
        pass

    pool.close()
    pool.join()
























