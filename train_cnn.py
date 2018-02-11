import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential

from keras.layers import Input, Flatten, Dense, Dropout, Conv1D, Conv2D, Activation, MaxPooling2D, BatchNormalization, Cropping2D
from sklearn.model_selection import train_test_split

def load_data():
    lines=[]
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    lines=lines[1:]
    images=[]
    measurements=[]
    for line in lines:
        # line = lines[l]
        source_path=line[0]
        file_name=source_path.split('/')[-1]
        current_path = './data/IMG/'+file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train=np.array(images)
    y_train=np.array(measurements)
    return X_train,y_train

def main(_):
    # == load data
    X_train, y_train = load_data()

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))
    print('-- nb_classes: '+str(nb_classes))
    input_shape = X_train.shape[1:]

    # == define model
    model = Sequential()
    model.add(Cropping2D(cropping=((55,25),(0,0)),input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(24,(5,5),border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(36,(5,5),border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(48,(5,5),border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64,(3,3),border_mode='same', activation='relu'))
    model.add(Conv2D(64,(3,3),border_mode='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    # train model
    model.fit(X_train, y_train, 100, 5,validation_data=(X_val, y_val), shuffle=True)
    model.save('model_nvidia.h5')  # creates a HDF5 file 'my_model.h5'


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 80, 320, 3)        12        
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 80, 320, 24)       1824      
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 40, 160, 24)       0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 40, 160, 36)       21636     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 20, 80, 36)        0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 20, 80, 48)        43248     
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 10, 40, 48)        0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 10, 40, 64)        27712     
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 10, 40, 64)        36928     
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 25600)             0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               2560100   
# _________________________________________________________________
# dense_2 (Dense)              (None, 50)                5050      
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                510       
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 2,697,031
# Trainable params: 2,697,025
# Non-trainable params: 6
# _________________________________________________________________
# Train on 5625 samples, validate on 2411 samples
# Epoch 1/5
# 5625/5625 [==============================] - 647s 115ms/step - loss: 0.0160 - val_loss: 0.0118
# Epoch 2/5
# 5625/5625 [==============================] - 696s 124ms/step - loss: 0.0105 - val_loss: 0.0088
# Epoch 3/5
# 5625/5625 [==============================] - 564s 100ms/step - loss: 0.0096 - val_loss: 0.0100
# Epoch 4/5
# 5625/5625 [==============================] - 588s 105ms/step - loss: 0.0097 - val_loss: 0.0084
# Epoch 5/5
# 5625/5625 [==============================] - 630s 112ms/step - loss: 0.0092 - val_loss: 0.0084
# swig/python detected a memory leak of type 'int64_t *', no destructor found.
