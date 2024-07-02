import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset(cfg, shuffle=True):
    if cfg['dataset'] == 'imagenette':
        (ds_train, ds_val, ds_test), ds_info = tfds.load('imagenette/320px-v2',
                                                         split=['train[:85%]', 'train[85%:]', 'validation'],
                                                         data_dir = '../tensorflow_datasets/',
                                                         shuffle_files=shuffle,
                                                         as_supervised=True,
                                                         with_info=True,)
        
    elif cfg['dataset'] == 'svhn':
        (ds_train, ds_val, ds_test), ds_info = tfds.load('svhn_cropped',
                                                         split=['train[:85%]', 'train[85%:]', 'test'],
                                                         data_dir = '../tensorflow_datasets/',
                                                         shuffle_files=shuffle,
                                                         as_supervised=True,
                                                         with_info=True,)
    
    elif cfg['dataset'] == 'celeb_a':
        (ds_train, ds_test), ds_info = tfds.load(cfg['dataset'],
                                       split=['train', 'validation'],
                                       data_dir = '../tensorflow_datasets/',
                                       with_info=True,)
    else:
        (ds_train, ds_val, ds_test), ds_info = tfds.load(cfg['dataset'],
                                                         split=['train[:85%]', 'train[85%:]', 'test'],
                                                         data_dir = '../tensorflow_datasets/',
                                                         shuffle_files=shuffle,
                                                         as_supervised=True,
                                                         with_info=True,)
    return ds_train, ds_val, ds_test, ds_info

def preprocess_dataset(dataset, cfg, n_classes=None, resize=False, normalize=False, onehot=False):
    if cfg['dataset'] == 'celeb_a':
        dataset = dataset.map(as_supervised)
        dataset = dataset.map(filter_keys)
    if resize:
        dataset = dataset.map(resize_image)
    if normalize:
        dataset = dataset.map(lambda x,y: normalize_img(x,y), num_parallel_calls=tf.data.AUTOTUNE)
    if onehot:
        dataset = dataset.map(lambda x,y: onehot_encode(x,y,n_classes,cfg), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def prefetch_dataset(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def onehot_encode(image, label, n_classes, cfg):
    if cfg['dataset'] == 'celeb_a':
        label = [int(value) for attribute, value in label.items()]
    else:
        label = tf.one_hot(label, depth=n_classes)
    return image, label

def resize_image(image, label):
    return tf.image.resize(image, (224, 224)), label

def as_supervised(dataset):
    image = dataset['image']
    attributes = dataset['attributes']
    return tf.dtypes.cast(image, tf.float32), attributes

def filter_keys(x, y):
    class_to_remove = ['Attractive', 'Blurry', 'Chubby', 'Male', 'Pale_Skin', 'Young', 'Heavy_Makeup', 'Oval_Face']
    filtered_y = {key: value for key, value in y.items() if key not in class_to_remove}
    return x, filtered_y
