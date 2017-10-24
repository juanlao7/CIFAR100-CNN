from keras.datasets import cifar100
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

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

# Defining the model.
model = Sequential()
model.add(Conv2D(12, (3, 3), activation=ACTIVATION, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (3, 3), activation=ACTIVATION))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(192, (3, 3), activation=ACTIVATION))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=ACTIVATION))
model.add(Dense(N_CLASSES, activation='softmax'))

model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model.
stopper = EarlyStopping(monitor='val_loss', patience=VALIDATION_PATIENCE)
h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[stopper], validation_split=VALIDATION_SPLIT)

# Evaluating the model.
score = model.evaluate(x_test, y_test, verbose=0)
print 'test_loss:', score[0]
print 'test_acc:', score[1]

result = [str(round(i, 4)) for i in [h.history['loss'][-1], h.history['acc'][-1], h.history['val_loss'][-1], h.history['val_acc'][-1], score[0], score[1]]]
print ','.join(result)
