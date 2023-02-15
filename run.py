import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

filenames = ['mod1', 'mod2', 'mod3', 'mod4', 'mod5']

# train_X.shape = (60000, 28, 28)
# train_y.shape = (60000,)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#flatten test_X to test_x (60000, 784)
#train_x = np.array([i.flatten() for i in train_X])
#test_x = np.array([i.flatten() for i in test_X])

#load model
def run(filename, test_x=test_x, test_y=test_y):
    model = tf.keras.models.load_model(filename)

    out = model.predict(test_x)

    pred = np.array([np.where(i==max(i))[0][0] for i in out])

    # test_pred = pred[10:]
    # actual = test_y[10:]
    # for i in zip(test_pred, actual):
    #     print(f'Predicted: {i[0]}   Actual: {i[1]}')

    mistakes = dict()
    count = 0
    for i in range(len(pred)):
        if pred[i] == test_y[i]:
            count += 1
        else:
            mistakes[test_y[i]] = pred[i]
            
    print(f'=={filename}==')
    print(f'=> {count} out of {len(test_y)}')
    print(f'=> Calculated accuracy: %{count/(len(test_y))*100}')

 for i in filenames:
     run(i)

