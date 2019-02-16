import matplotlib.pyplot as plt
import numpy as np

def visualize_history(history):
    train_acc  = history.history['acc']
    val_acc    = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']

    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    ax.set_ylim([0.5,1])
    ax.grid(False)
    ax.plot(train_acc, label='train_acc', color='g')
    ax.plot(val_acc, label='val_acc', color='r')
    ax.set_xlabel('Epochs', fontsize=17)
    ax.set_ylabel('Accuracy', fontsize=17)
    ax.set_title('Accuracy Graph')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    ax.set_ylim([0,1])
    ax.grid(False)
    ax.plot(train_loss, label='train_loss', color='g')
    ax.plot(val_loss, label='val_loss', color='r')
    ax.set_xlabel('Epochs', fontsize=17)
    ax.set_ylabel('Loss', fontsize=17)
    ax.set_title('Loss Graph')
    plt.legend()
    plt.show()
    return True

def get_conf_matrix(model, val, n_classes=5):
    conf_matrix = np.zeros((n_classes,n_classes),dtype=int)
    val_x, val_y = val
    VAL_Y = np.asarray(val_y, dtype=int)
    preds_ind, wrong_preds_ind = list(), list()
    for i in range(len(val_x)):
        pred = model.predict(np.expand_dims(val_x[i], axis=0))
        maxi = np.amax(pred)
        ind  = np.where(pred==maxi)[1][0]
        conf_matrix[VAL_Y[i],ind]+=1
        preds_ind.append(ind)
        if VAL_Y[i]!=ind:
            wrong_preds_ind.append(i)

    return conf_matrix, preds_ind, wrong_preds_ind

def visualize_conf_matrix(conf_matrix, class_dict=None):
    plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.matshow(conf_matrix, cmap='Blues')
    ax.set_xlabel('PREDICTIONS', fontsize=17)
    ax.set_ylabel('GROUNDTRUTH', fontsize=17)
    if class_dict != None:
        ax.set_xticklabels(['']+list(class_dict.values()), fontsize=15)
        ax.set_yticklabels(['']+list(class_dict.values()), fontsize=15)
    for (i, j), z in np.ndenumerate(conf_matrix):
        if z > 50: 
            color = 'white'
        else:
            color = 'black'
        ax.text(j, i, '{}'.format(z), ha='center', va='center', fontsize=18, color=color)
    plt.show()
    return True

def visualize_wrong_preds(wrong_preds_ind, val, preds, class_dict=None):
    val_x, val_y = val
    VAL_Y = np.asarray(val_y, dtype=int)
    for i in wrong_preds_ind:
        plt.figure()
        ax = plt.subplot(111)
        if class_dict != None:
            pred = class_dict[preds[i]]
            gt   = class_dict[VAL_Y[i]]
        else:
            pred = preds[i]
            gt   = VAL_Y[i]
        ax.set_title('Prediction: {} - Groundtruth: {}'.format(pred, gt))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(np.asarray(val_x[i,...,::-1], dtype=int))
        ax.grid(False)
        plt.show()
    return True
  
