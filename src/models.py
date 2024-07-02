import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import vgg16, resnet_v2, inception_v3, densenet
import pickle
import importlib

def get_optimizer(optimizer_name, learning_rate):
    optimizer_module = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(optimizer_module, optimizer_name)
    optimizer = optimizer_class(learning_rate=learning_rate)
    return optimizer

def mlp(dataset, n_layers=3, n_hidden=512):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(dataset.element_spec[0].shape[1:])))
    for i in range(n_layers):
        model.add(Dense(n_hidden, activation='relu'))
    model.add(Dense(dataset.element_spec[1].shape[1], activation='softmax'))
    return model

def cnn(dataset):
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(dataset.element_spec[0].shape[1:])))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(dataset.element_spec[1].shape[1], activation='softmax'))
    return model

def VGG16(dataset,cfg):
    base_model = vgg16.VGG16(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    if cfg['dataset'] == 'celeb_a':
        outputs = Dense(dataset.element_spec[1].shape[1], activation='sigmoid')(x)
    else:
        outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def pretrained_VGG16(dataset,cfg):
    base_model = vgg16.VGG16(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    if cfg['dataset'] == 'celeb_a':
        outputs = Dense(dataset.element_spec[1].shape[1], activation='sigmoid')(x)
    else:
        outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def pretrained_VGG16_v2(dataset,info):
    base_model = vgg16.VGG16(include_top=False,input_shape=((224,224,3)))
    for layer in base_model.layers[:-2]:
        layer.trainable=False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    if info['dataset'] == 'celeb_a':
        outputs = Dense(dataset.element_spec[1].shape[1], activation='sigmoid')(x)
    else:
        outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def ResNet50V2(dataset):
    base_model = resnet_v2.ResNet50V2(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def pretrained_ResNet50V2(dataset):
    base_model = resnet_v2.ResNet50V2(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def InceptionV3(dataset):
    base_model = inception_v3.InceptionV3(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def pretrained_InceptionV3(dataset):
    base_model = inception_v3.InceptionV3(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def DenseNet121(dataset):
    base_model = densenet.DenseNet121(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def pretrained_DenseNet121(dataset):
    base_model = densenet.DenseNet121(include_top=False,input_shape=(dataset.element_spec[0].shape[1:]))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(dataset.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def train_model(ds_train, ds_val, cfg, exp_name=None, save=False):
    if cfg['model'] == 'mlp':
        model = mlp(ds_train)
    elif cfg['model'] == 'cnn':
        model = cnn(ds_train)
    elif cfg['model'] == 'vgg16':
        model = VGG16(ds_train,cfg)
    elif cfg['model'] == 'pretrained_vgg16':
        model = pretrained_VGG16(ds_train, cfg)
    elif cfg['model'] == 'pretrained_vgg16_v2':
        model = pretrained_VGG16_v2(ds_train, info)
    elif cfg['model'] == 'resnet50':
        model = ResNet50V2(ds_train)
    elif cfg['model'] == 'pretrained_resnet50':
        model = pretrained_ResNet50V2(ds_train)
    elif cfg['model'] == 'inception':
        model = InceptionV3(ds_train)
    elif cfg['model'] == 'pretrained_inception':
        model = pretrained_InceptionV3(ds_train)
    elif cfg['model'] == 'densenet121':
        model = DenseNet121(ds_train)
    elif cfg['model'] == 'pretrained_densenet121':
        model = pretrained_DenseNet121(ds_train)
    else:
        raise NotImplementedError(f"Model ({cfg['model']}) not implemented.")
    model.save(exp_name+'/untrained_model.keras')
    
    optimizer = get_optimizer(cfg['optimizer'], cfg['learning_rate'])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=cfg['patience'], restore_best_weights=True)
    history = model.fit(ds_train, verbose=1, epochs=cfg['max_epoch'], validation_data=ds_val, callbacks=[early_stop,])
    if save:
        model.save(exp_name+'/trained_model.keras')
        with open(exp_name+'/history.pickle', 'wb') as f:
            pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model, history
