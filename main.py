import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import segmentation_models as sm

from train import dataset, aug


BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['car', 'pedestrian']
LR = 0.0001
EPOCHS = 40

PATHS = {
    'x_train' : './CamVid/train/',
    'y_train' : './CamVid/trainannot/',
    'x_val'   : './CamVid/val/',
    'y_val'   : './CamVid/valannot/',
    'x_test'  : './CamVid/test/',
    'y_test'  : './CamVid/testannot/'
    }


def load_data():
    x_train = dataset.read_dataset_dir(PATHS['x_train'], flag=0)
    y_train = dataset.read_dataset_dir(PATHS['y_train'], flag=1)

    x_val = dataset.read_dataset_dir(PATHS['x_val'], flag=0)
    y_val = dataset.read_dataset_dir(PATHS['y_val'], flag=1)

    return x_train, y_train, x_val, y_val


def load_dataset():
    train_ds = dataset.Dataset(PATHS['x_train'], PATHS['y_train'], classes=CLASSES)
    val_ds = dataset.Dataset(PATHS['x_val'], PATHS['y_val'], classes=CLASSES)
    
    train_dl = dataset.Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = dataset.Dataloder(val_ds, batch_size=1, shuffle=True)

    return train_dl, val_dl


def main():
    #x_train, y_train, x_val, y_val = load_data()
    #train_dataloader, valid_dataloader = load_dataset()

    preprocess_input = sm.get_preprocessing(BACKBONE)

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

    # load trained model
    model.load_weights('best_model.h5') 

    # Dataset for train images
    test_dataset = dataset.Dataset(
        PATHS['x_test'], 
        PATHS['y_test'], 
        classes=CLASSES, 
        augmentation=aug.get_training_augmentation(),
        preprocessing=aug.get_preprocessing(preprocess_input),
    )
    test_dataloader = dataset.Dataloder(test_dataset, batch_size=1, shuffle=False)

    scores = model.evaluate(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))
    
    id = np.random.choice(np.arange(len(test_dataset)))

    image, gt_mask = test_dataset[id]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    plt.subplot(131)
    plt.imshow(image[0,:,:,:])
    plt.subplot(132)
    plt.imshow(gt_mask)
    plt.subplot(133)
    plt.imshow(pr_mask[0,:,:,:])
    plt.show()


if __name__ == '__main__':
    main()

