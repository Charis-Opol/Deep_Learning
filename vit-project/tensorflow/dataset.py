"""CIFAR-10 loaders: plain (no augmentation) and augmented variants.

Reads the standard Python-pickle CIFAR-10 batches directly from disk (the same files
torchvision downloads for the PyTorch side) instead of tf.keras.datasets.cifar10 /
tensorflow_datasets, both of which re-download a separate copy from a slow mirror.
"""
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _load_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW -> NHWC
    labels = np.array(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir="../../data/cifar-10-batches-py"):
    data_dir = Path(data_dir)
    train_images, train_labels = [], []
    for i in range(1, 6):
        images, labels = _load_batch(data_dir / f"data_batch_{i}")
        train_images.append(images)
        train_labels.append(labels)
    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    x_test, y_test = _load_batch(data_dir / "test_batch")
    return (x_train, y_train), (x_test, y_test)


def _normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD
    return image, label


def _augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


def get_dataloaders(data_dir="../../data/cifar-10-batches-py", batch_size=128, augment=False, subset_size=None):
    """subset_size: if set, restrict the train set to this many images (for quick smoke tests)."""
    (x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)

    if subset_size is not None:
        x_train, y_train = x_train[:subset_size], y_train[:subset_size]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10_000).batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(x_train) // batch_size
    return train_ds, test_ds, steps_per_epoch


if __name__ == "__main__":
    train_ds, test_ds, steps_per_epoch = get_dataloaders(augment=False, subset_size=256)
    for images, labels in train_ds.take(1):
        print(f"batch images: {tuple(images.shape)}, labels: {tuple(labels.shape)}")
    print(f"steps per epoch: {steps_per_epoch}")
