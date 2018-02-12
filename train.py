import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle
from keras.models import load_model

def load_data():
    import csv
    lines=[]
    with open('./run_round/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    shuffle(lines)
    images=[]
    measurements=[]
    for line in lines:
        source_path=line[0]
        file_name=source_path.split('/')[-1]
        current_path = './run_round/IMG/'+file_name
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
    print(X_train.shape, y_train.shape)

    for epoch in range(0,10):
        model = load_model('model'+str(epoch)+'.h5')
        model.fit(X_train,y_train,batch_size=100, epochs=1,validation_split=0.1, shuffle=True, callbacks=None)
        model.save('model'+str(epoch+1)+'.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

