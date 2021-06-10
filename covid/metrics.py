import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
sns.set(font_scale=1.5, style='white')

from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, average_precision_score

import tensorflow as tf
import tensorflow_addons as tfa

from .utils import CLASS_NAMES


def f1_metric(num_classes, average='weighted', name='f1_score'):
    f1_metric = tfa.metrics.F1Score(num_classes=num_classes, average=average, name=name)
    
    return f1_metric


def plot_history(history, title, legend='upper right'):
    epochs_range = range(len(history['accuracy']))
    
    plt.figure(figsize=(35, 8))
    plt.suptitle(title, fontsize=20)
    plt.subplot(131)
    plt.plot(epochs_range, history['accuracy'], label='Train')
    plt.plot(epochs_range, history['val_accuracy'], label='Val')
    plt.legend(loc=legend)
    plt.title('Accuracy')
    plt.ylim(0,1)

    plt.subplot(132)
    plt.plot(epochs_range, history['loss'], label='Train')
    plt.plot(epochs_range, history['val_loss'], label='Val')
    plt.legend(loc=legend)
    plt.title('Loss')
    plt.yscale('log')

    plt.subplot(133)
    plt.plot(epochs_range, history['f1_score'], label='Train')
    plt.plot(epochs_range, history['val_f1_score'], label='Val')
    plt.legend(loc=legend)
    plt.title('f1 score')
    plt.ylim(0,1);
    
    
def plot_from_json(path, title):
    with open(path) as f:
        history = json.load(f)
        
    plot_history(history[0], title)
    
    
def classification_metrics(model, X_val, y_val, normalize=None, show_report=True, fmt='d'):
    y_pred = model.predict(X_val)
    
    if show_report:
        f1_metric = tfa.metrics.F1Score(num_classes=3, average='weighted', name='f1_score')
        f1_metric.update_state(y_val, y_pred)
        result = f1_metric.result()
        print('F1 SCORE: ', result.numpy())

        print(classification_report(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1), target_names=CLASS_NAMES, zero_division=0))
        
    if normalize:
        fmt = '.2f'

    cm = confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1), normalize=normalize)
#     plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, fmt=fmt)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=0)
    plt.yticks(va='center');


def show_both_matrices(model, model_name, X_val, y_val, config, images, title):
    plt.figure(figsize=(20,9))
    plt.subplot(121)
    classification_metrics(model, X_val, y_val)
    plt.title(model_name + ' - ' + images + config)
    plt.subplot(122)
    classification_metrics(model, X_val, y_val, normalize='true', show_report=False)
    plt.title(title)
    plt.show();
    

def confusion_matrices(model_name, get_model, get_model_dataaug, filename, X_val, y_val, images, title):
    model = get_model
    model.load_weights('best-models/' + model_name.lower() + '-weights/' + filename + '.ckpt')
    show_both_matrices(model, model_name, X_val, y_val, '', images, title)
    
    model = get_model
    model.load_weights('best-models/' + model_name.lower() + '-weights/' + filename + '-classw.ckpt')
    show_both_matrices(model, model_name, X_val, y_val, ' + Class Weight', images, title)
    
    model = get_model_dataaug
    model.load_weights('best-models/' + model_name.lower() + '-weights/' + filename + '-dataaug.ckpt')
    show_both_matrices(model, model_name, X_val, y_val, ' + Data Augmentation', images, title)
    
    model = get_model_dataaug
    model.load_weights('best-models/' + model_name.lower() + '-weights/' + filename + '-classw-dataaug.ckpt')
    show_both_matrices(model, model_name, X_val, y_val, ' + Class Weight + Data Augmentation', images, title)   
    
    
def plot_roc_curves(model, X_test, y_test):

    labels = ['COV', 'Normal', 'OtherPneumonia']

    y_pred = model.predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {0:0.4f})'.format(roc_auc["micro"]))

    for i in range(3):
        plt.plot(fpr[i], tpr[i], label='{0} (AUC = {1:0.4f})'.format(labels[i], roc_auc[i]))

    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    
    
def plot_pr_curves(model, X_test, y_test):

    labels = ['COV', 'Normal', 'OtherPneumonia']

    y_pred = model.predict(X_test)

    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], y_pred[:,i])
        avg_precision[i] = average_precision_score(y_test[:,i], y_pred[:,i])

    precision['micro'], recall['micro'], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    avg_precision["micro"] = average_precision_score(y_test.ravel(), y_pred.ravel())

    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.05])

    plt.plot(recall['micro'], precision['micro'], label='micro-average PR curve (AP = {0:0.4f})'.format(avg_precision['micro']))

    for i in range(len(labels)):
        plt.plot(recall[i], precision[i], label='{0} (AP = {1:0.4f})'.format(labels[i], avg_precision[i]))

    plt.legend(loc=3, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    
    
def print_auc(model, model_name, file_name, X_val, y_val, batch_size=32):
    model.load_weights('best-models/' + model_name.lower() + '-weights/' + file_name + '.ckpt')
    y_pred = model.predict(X_val, batch_size=batch_size)
    fpr, tpr, _ = roc_curve(y_val.ravel(), y_pred.ravel())

    print(model_name, auc(fpr, tpr))
    
    
def calculate_aucs(filename, model_functions, model_functions_rgb, model_bit, X_val, y_val, X_val_rgb, y_val_rgb, batch_size):
    for model_name, model in zip(['Simple', 'Tiny', 'Small', 'LargeW', 'LargeT'], model_functions):
        print_auc(model, model_name, filename, X_val, y_val, batch_size)
    
    for model_name, model in zip(['EfficientNetB3-ImageNet', 'EfficientNetB3'], model_functions_rgb):
        print_auc(model, model_name, filename, X_val_rgb, y_val_rgb, batch_size)

    print_auc(model_bit, 'GoogleBiT', filename, X_val_rgb/np.max(X_val_rgb), y_val_rgb)
    
    
def plot_f1_scores(f1):
    for i in range(len(f1)):
        plt.figure(figsize=(11,7))
        
        for j in range(0, len(f1[i]), 4):
            plt.bar(j-1.05, f1[i][j], width=0.7, linewidth=0, label='baseline', color=sns.color_palette('deep')[0])
            plt.bar(j-0.35, f1[i][j+1], width=0.7, linewidth=0, label='classw', color=sns.color_palette('deep')[2])
            plt.bar(j+0.35, f1[i][j+2], width=0.7, linewidth=0, label='dataaug', color=sns.color_palette('Set2')[6])
            plt.bar(j+1.05, f1[i][j+3], width=0.7, linewidth=0, label='classw-dataaug', color=sns.color_palette('deep')[4])
    
        plt.ylabel('f1_score')
        plt.yticks(np.arange(0,1.1,step=0.1))
        plt.xticks(range(0,20,4), ['original', 'nobackg', 'crop', 'lungs-nocrop', 'lungs'])
        plt.legend(['baseline', 'classw', 'dataaug', 'classw-dataaug'], bbox_to_anchor=(1,0.5), loc='center left')
        plt.ylim(0,1)
        plt.title(models[i]);