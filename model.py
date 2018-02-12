from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Conv1D, Conv2D, Activation, MaxPooling2D, BatchNormalization, Cropping2D, Lambda

# == define model
def converter(x):
    # x has shape (batch, width, height, channels)
    # luminosity formula
    # Normalized
    return ( (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:]) ) / 128.0 - 1.0

model = Sequential()
input_shape=(160,320,3)
model.add(Cropping2D(cropping=((25,25),(0,0)),input_shape=input_shape))
model.add(Lambda(converter))
# model.add(BatchNormalization())
model.add(Conv2D(8,(5,5),strides=(1,1),padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,(5,5),strides=(1,1),padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(24,(5,5),strides=(1,1),padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(5,5),padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Conv2D(40,(3,3),padding='same', activation='relu'))
model.add(Conv2D(40,(3,3),padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.save('model0.h5')
