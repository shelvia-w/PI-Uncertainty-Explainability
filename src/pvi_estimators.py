import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense

from src.models import get_optimizer
from src.datasets import prefetch_dataset
from src.utils import softmax

def train_pvi_null_model(dataset, model, cfg, save_path):
    ds_null = dataset.map(lambda x, y: (tf.zeros_like(x), y))
    ds_null = prefetch_dataset(ds_null, batch_size=cfg['batch_size'])
    
    optimizer = get_optimizer(cfg['optimizer'], cfg['learning_rate'])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001, restore_best_weights=True)
    model.fit(ds_null, verbose=1, epochs=50, callbacks=[early_stop,])
    model.save(save_path)

def train_pvi_model_from_scratch(ds_train, ds_val, model, cfg, save_path):
    ds_train = prefetch_dataset(ds_train, batch_size=cfg['batch_size'])
    ds_val = prefetch_dataset(ds_val, batch_size=cfg['batch_size'])
    
    optimizer = get_optimizer(cfg['optimizer'], cfg['learning_rate'])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(ds_train, verbose=1, epochs=cfg['max_epoch'], validation_data=ds_val, callbacks=[early_stop,])
    model.save(save_path)

def v_entropy(x,y,model):
    prob = model.predict(x)
    return -1 * np.log2(tf.boolean_mask(prob, y).numpy())

def v_entropy_ensemble(x1,x2,y,model1,model2):
    prob1 = model1.predict(x1)
    prob2 = model2.predict(x2)
    avg_prob = (prob1 + prob2) / 2
    return -1 * np.log2(tf.boolean_mask(avg_prob, y).numpy())

def neural_pvi(x,y,model,null_model):
    null_x = np.zeros_like(x)
    v_cond_entropy = v_entropy(x,y,model)
    v_null_entropy = v_entropy(null_x,y,null_model)
    pvi = v_null_entropy - v_cond_entropy
    return pvi

def neural_pvi_ensemble(x1,x2,y,model1,model2,null_model1,null_model2):
    null_x1 = np.zeros_like(x1)
    null_x2 = np.zeros_like(x2)
    v_cond_entropy = v_entropy_ensemble(x1,x2,y,model1,model2)
    v_null_entropy = v_entropy_ensemble(null_x1,null_x2,y,null_model1,null_model2)
    pvi = v_null_entropy - v_cond_entropy
    return pvi

def v_entropy_calibrated(x,y,model,temp):
    logits_layer = model.layers[-1]
    logits_layer.activation = None 
    logits_model = tf.keras.models.Model(inputs=model.input, outputs=logits_layer.output)
    preds = logits_model.predict(x)
    prob = np.array([softmax(x/temp) for x in preds])
    return -1 * np.log2(tf.boolean_mask(prob, y).numpy())

def neural_pvi_calibrated(x,y,model,null_model,model_temp,null_temp):
    null_x = np.zeros_like(x)
    v_cond_entropy = v_entropy_calibrated(x,y,model,model_temp)
    v_null_entropy = v_entropy_calibrated(null_x,y,null_model,null_temp)
    pvi = v_null_entropy - v_cond_entropy
    return pvi