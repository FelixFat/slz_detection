import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import segmentation_models as sm

from train import dataset, viz


BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['car', 'pedestrian']
LR = 0.0001
EPOCHS = 40

#CLASSES = [
#    'sky', 'building', 'pole', 'road', 'pavement',
#    'tree', 'signsymbol', 'fence', 'car',
#    'pedestrian', 'bicyclist', 'unlabelled'
#    ]

# https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing
PATHS = {
    'train_images' : './sample-dataset/images_prepped_train/',
    'train_masks' : './sample-dataset/annotations_prepped_train/',
    'test_images' : './sample-dataset/images_prepped_test/',
    'test_masks' : './sample-dataset/annotations_prepped_test/'
    }


def load_data():
    #train_ds = dataset.Dataset(PATHS['train_images'], PATHS['train_masks'], classes=CLASSES)
    #test_ds = dataset.Dataset(PATHS['train_images'], PATHS['train_masks'], classes=CLASSES)
    
    #train_dl = dataset.Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    #test_dl = dataset.Dataloder(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    x_train = dataset.read_dataset_dir(PATHS['train_images'], flag=0)
    y_train = dataset.read_dataset_dir(PATHS['train_masks'], flag=1)

    x_test = dataset.read_dataset_dir(PATHS['test_images'], flag=0)
    y_test = dataset.read_dataset_dir(PATHS['test_masks'], flag=1)

    return x_train, y_train, x_test, y_test


def load_dataset():
    train_ds = dataset.Dataset(PATHS['train_images'], PATHS['train_masks'], classes=CLASSES)
    test_ds = dataset.Dataset(PATHS['train_images'], PATHS['train_masks'], classes=CLASSES)
    
    train_dl = dataset.Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = dataset.Dataloder(test_ds, batch_size=1, shuffle=True)

    return train_dl, test_dl


def main():
    x_train, y_train, x_test, y_test = load_data()
    train_dataloader, valid_dataloader = load_dataset()

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
    )


if __name__ == '__main__':
    main()

