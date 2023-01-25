import os

import cv2
import numpy as np

from tensorflow import keras


def read_img_color(path, shape=None, norm=False):
    """Read color image from path"""

    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    if shape is not None:
        im = cv2.resize(im, shape)
    
    im = im.astype(np.float32)
    if norm is True:
        im = im / 255.0
    
    return im


def read_img_binary(path, shape=None, norm=False):
    """Read binary image from path"""

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if shape is not None:
        im = cv2.resize(im, shape)
    
    im = im.astype(np.float32)
    if norm is True:
        im = im / 255.0

    return im


def get_img_paths(dir_path):
    """Get images paths"""

    file_names = os.listdir(dir_path)

    return [os.path.join(dir_path, file) for file in file_names]


def read_dataset_dir(dir_path, shape=None, norm=True, flag=0):
    """Read dataset images from dir"""
    
    image_paths = get_img_paths(dir_path)

    img_dataset = np.array([])
    if flag == 0:
        img_dataset = np.array([read_img_color(img, shape, norm) for img in image_paths])
    elif flag == 1:
        img_dataset = np.array([read_img_binary(img, shape, norm) for img in image_paths])

    return img_dataset


class Dataset:
    """
    Create dataset from imgages
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = [
        'sky', 'building', 'pole', 'road', 'pavement',
        'tree', 'signsymbol', 'fence', 'car',
        'pedestrian', 'bicyclist', 'unlabelled'
        ]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        #self.class_values = {key : val for val, key in enumerate(classes)}
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values.values()]
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class Dataloder(keras.utils.Sequence):
    """
    Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = [self.dataset[i] for i in range(start, stop)]
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""

        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""

        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes) 

