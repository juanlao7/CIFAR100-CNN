from keras.datasets import cifar100
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import regularizers
import matplotlib.pyplot as plt
import os.path
import pickle

N_CLASSES = 100
SAMPLE_WIDTH = 32
SAMPLE_HEIGHT = 32

# Parameters
BATCH_SIZE = 100
N_EPOCHS = 10000                # We stop training when the validation loss converges; the training can take all the epochs it needs
VALIDATION_SPLIT = 0.2
VALIDATION_PATIENCE = 15
ACTIVATION = 'elu'
DROPOUT = 0.25

def plotOptions(results, title, ylabel, keys):
    plt.gca().set_color_cycle(None)
    plt.plot([], '--', color='black')
    plt.plot([], color='black')
    
    for i in results:
        plt.plot(results[i]['h'][keys[0]])
    
    plt.legend(['Training', 'Validation'] + results.keys(), loc='upper right')
    
    plt.gca().set_color_cycle(None)
    
    for i in results:
        plt.plot(results[i]['h'][keys[1]], '--')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.show()

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Normalizing the input.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

# One-hot encoding the labels.
y_train = np_utils.to_categorical(y_train, N_CLASSES)
y_test = np_utils.to_categorical(y_test, N_CLASSES)

# Reshaping the samples depending on which format the backend uses.

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, SAMPLE_WIDTH, SAMPLE_HEIGHT)
    x_test = x_test.reshape(x_test.shape[0], 3, SAMPLE_WIDTH, SAMPLE_HEIGHT)
    input_shape = (3, SAMPLE_WIDTH, SAMPLE_HEIGHT)
else:
    x_train = x_train.reshape(x_train.shape[0], SAMPLE_WIDTH, SAMPLE_HEIGHT, 3)
    x_test = x_test.reshape(x_test.shape[0], SAMPLE_WIDTH, SAMPLE_HEIGHT, 3)
    input_shape = (SAMPLE_WIDTH, SAMPLE_HEIGHT, 3)

#optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

weightDecayFunctions = {
    '0': None,
    '0.01': regularizers.l2(0.01),
    '0.001': regularizers.l2(0.001),
    '0.0001': regularizers.l2(0.0001),
    '0.00001': regularizers.l2(0.00001)
}

results = {}

for i in weightDecayFunctions:
    print '###    Weight decay factor ' + i + '    ###'
    
    resultFileName = 'model_' + i + '.result'
    
    if os.path.isfile(resultFileName):
        handler = open(resultFileName, 'rb')
        results[i] = pickle.load(handler)
        handler.close()
        continue
        
    # Defining the model.
    model = Sequential()
    model.add(Conv2D(27, (3, 3), activation=ACTIVATION, input_shape=input_shape, kernel_regularizer=weightDecayFunctions[i]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))
    model.add(Conv2D(81, (3, 3), activation=ACTIVATION, kernel_regularizer=weightDecayFunctions[i]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))
    model.add(Conv2D(135, (3, 3), activation=ACTIVATION, kernel_regularizer=weightDecayFunctions[i]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(128, activation=ACTIVATION, kernel_regularizer=weightDecayFunctions[i]))
    model.add(Dropout(DROPOUT))
    model.add(Dense(128, activation=ACTIVATION, kernel_regularizer=weightDecayFunctions[i]))
    model.add(Dense(N_CLASSES, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training the model.
    stopper = EarlyStopping(monitor='val_loss', patience=VALIDATION_PATIENCE)
    h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[stopper], validation_split=VALIDATION_SPLIT)
    
    # Evaluating the model.
    score = model.evaluate(x_test, y_test, verbose=0)
    
    results[i] = {
        'h': h.history,
        'test_loss': score[0],
        'test_acc': score[1]
    }
    
    handler = open(resultFileName, 'wb')
    pickle.dump(results[i], handler)
    handler.close()

print '### FINISH! ###'

for i in results:
    h = results[i]['h']
    print i + ':'
    result = [str(round(i, 4)) for i in [h['loss'][-1], h['acc'][-1], h['val_loss'][-1], h['val_acc'][-1], results[i]['test_loss'], results[i]['test_acc']]]
    print ','.join(result)
    
# Plotting
plotOptions(results, 'Model loss', 'Loss', ['val_loss', 'loss'])
plotOptions(results, 'Model accuracy', 'Accuracy', ['val_acc', 'acc'])
