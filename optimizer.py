from keras.datasets import cifar100
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import matplotlib.pyplot as plt

N_CLASSES = 100
SAMPLE_WIDTH = 32
SAMPLE_HEIGHT = 32

# Parameters    
BATCH_SIZE = 100
N_EPOCHS = 10000                # We stop training when the validation loss converges; the training can take all the epochs it needs
VALIDATION_SPLIT = 0.2
VALIDATION_PATIENCE = 15
ACTIVATION = 'elu'

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

optimizers = {
    'SGD': SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
    'RMSProp': RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    'Adagrad': Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
    'Adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
    'Adam': Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    'Adamax': Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    'Nadam': Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
}

results = {}

for i in optimizers:
    print '###    Optimizer ' + i + '    ###'
    
    # Defining the model.
    model = Sequential()
    model.add(Conv2D(27, (3, 3), activation=ACTIVATION, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(81, (3, 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(135, (3, 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=ACTIVATION))
    model.add(Dense(128, activation=ACTIVATION))
    model.add(Dense(N_CLASSES, activation='softmax'))
    
    model.compile(optimizer=optimizers[i], loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training the model.
    stopper = EarlyStopping(monitor='val_loss', patience=VALIDATION_PATIENCE)
    h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[stopper], validation_split=VALIDATION_SPLIT)
    
    # Evaluating the model.
    score = model.evaluate(x_test, y_test, verbose=0)
    
    results[i] = {
        'h': h,
        'test_loss': score[0],
        'test_acc': score[1]
    }
    
print '### FINISH! ###'

for i in optimizers:
    h = results[i]['h']
    print i + ':'
    result = [str(round(i, 4)) for i in [h.history['loss'][-1], h.history['acc'][-1], h.history['val_loss'][-1], h.history['val_acc'][-1], results[i]['test_loss'], results[i]['test_acc']]]
    print ','.join(result)
    
# Plotting

plt.gca().set_color_cycle(None)

for i in optimizers:
    plt.plot(results[i]['h'].history['val_loss'])

plt.plot([], '--', color='black')
plt.legend(optimizers.keys() + ['Training loss'], loc='upper right')

plt.gca().set_color_cycle(None)

for i in optimizers:
    plt.plot(results[i]['h'].history['loss'], '--')

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

