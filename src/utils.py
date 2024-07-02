import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def temp_scaling(model, ds, iters=300):
    logits_layer = model.layers[-1]
    logits_layer.activation = None 
    logits_model = tf.keras.models.Model(inputs=model.input, outputs=logits_layer.output)
    
    y_pred = logits_model.predict(ds.batch(512))
    num_bins = 50
    y = [y.numpy().astype(np.int32) for x,y in ds]
    y = np.argmax(y, axis=1)
    labels_true = tf.convert_to_tensor(y, dtype=tf.int32, name='labels_true')
    logits = tf.convert_to_tensor(y_pred, dtype=tf.float32, name='logits')
    
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32) 

    def compute_temp_loss():
        y_pred_model_w_temp = tf.math.divide(y_pred, temp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.convert_to_tensor(tf.keras.utils.to_categorical(y)), y_pred_model_w_temp))
        return loss

    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    for i in range(iters):
        opts = optimizer.minimize(compute_temp_loss, var_list=[temp])

    opt_temp = temp.numpy()
    return opt_temp

def compute_opt_threshold(metric, true_label):
    sorted_metric = np.sort(metric)
    threshold_list = []
    filtering_acc_list = []
    for i in range(len(metric)):
        threshold = sorted_metric[i]
        threshold_list.append(threshold)
        metric_label = [1 if value >= threshold else 0 for value in metric]
        correct_samples = sum(1 for true, metric in zip(true_label, metric_label) if true == metric)
        filtering_acc = (correct_samples / len(true_label)) * 100
        filtering_acc_list.append(filtering_acc)
    idx = np.argmax(filtering_acc_list)
    opt_threshold = threshold_list[idx]
    return opt_threshold

def compute_filtering_acc(metric, true_label, threshold):
    metric_label = [1 if value >= threshold else 0 for value in metric]
    correct_samples = sum(1 for true, metric in zip(true_label, metric_label) if true == metric)
    filtering_acc = (correct_samples / len(true_label)) * 100
    return filtering_acc

def reliability_diagram(metric, true_y, pred_y, n_bins):
    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(metric, bins, right=True)

    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_acc[b] = np.mean(true_y[selected] == pred_y[selected])
            bin_conf[b] = np.mean(metric[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_acc * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_conf * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_acc - bin_conf)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts) * 100
    
    result = {
        'bins' : bins,
        'bin_size' : bin_size,
        'bin_counts' : bin_counts,
        'bin_acc' : bin_acc,
        'bin_conf' : bin_conf,
        'avg_acc' : avg_acc,
        'avg_conf' : avg_conf,
        'gaps' : gaps,
        'ece' : ece
    }
    
    return result

def plot_reliability_diagram(result, metric):
    positions = result['bins'][:-1] + result['bin_size']/2.0

    fig, axs = plt.subplots(2, 1, figsize=(4,5), dpi=100, sharex=True, gridspec_kw={'height_ratios': [1,0.5]})

    axs[0].bar(positions,
            result['bin_acc'],
            width=result['bin_size'],
            edgecolor='black',
            color='blue',
            linewidth=1,
            label='Accuracy')

    axs[0].bar(positions,
            result['gaps'],
            bottom=np.minimum(result['bin_acc'], result['bin_conf']),
            width=result['bin_size'],
            edgecolor='black',
            color='red',
            linewidth=1,
            hatch="//",
            label='Gap')

    axs[0].text(0.7, 0.1,
                f"ECE={result['ece']:.2f}",
                color="black",
                bbox=dict(facecolor='white', alpha=0.5),
                fontsize=11)

    axs[0].plot([0,1], [0,1], linestyle = "--", color="gray")
    axs[0].set_xlim(0,1)
    axs[0].grid(True, alpha=0.3)
    axs[0].grid(zorder=0)
    axs[0].set_ylabel('Accuracy', fontsize=11)
    axs[0].legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    axs[1].bar(positions,
               result['bin_counts'] / np.sum(result['bin_counts']),
               width=result['bin_size'],
               edgecolor='black',
               color='tab:orange')

    axs[1].axvline(x=result['avg_acc'], linestyle="--", color="blue", label='Average Accuracy')
    axs[1].axvline(x=result['avg_conf'], linestyle='--', color='red', label='Average Confidence')

    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(0.0, 1.0, 11))
    axs[1].set_xlabel(metric, fontsize=11)
    axs[1].set_ylabel('% of Samples', fontsize=11)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
def compute_ece(metric, true_y, pred_y, n_bins):
    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(metric, bins, right=True)

    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_acc[b] = np.mean(true_y[selected] == pred_y[selected])
            bin_conf[b] = np.mean(metric[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_acc - bin_conf)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts) * 100
    return ece

def compute_nll(metric, true_y, n_classes):
    cce = tf.keras.losses.CategoricalCrossentropy()
    true_y = tf.one_hot(true_y, depth=n_classes)
    return cce(true_y, metric).numpy()

def compute_brier_score(metric, true_y, n_classes):
    true_y = tf.one_hot(true_y, depth=n_classes)
    brier_score = np.mean(np.sum((metric - true_y)**2, axis=1))
    return brier_score

def compute_classification_error(metric, true_y):
    count = 0
    for i in range(len(metric)):
        pred = np.argmax(metric[i])
        if true_y[i] == pred:
            count += 1
    error = 1 - count/len(metric)
    return error

def compute_selective_pred_acc(metric, dataset, model, n):
    x = np.array([x for x, y in dataset])
    y = np.array([y for x, y in dataset])
    idx = np.argsort(metric)[n:]
    eval_ds = model.evaluate(x[idx], y[idx], verbose=0)
    return eval_ds[1]