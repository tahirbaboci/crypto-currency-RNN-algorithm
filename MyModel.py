# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint # ModelCheckpoint always saves the best one from every epochs


SEQ_LEN = 60 # use last 60 minute of pricing data
FUTURE_PERIOD_PREDICT = 3 # next minutes
RATIO_TO_PREDICT = "LTC-USD" # which one we are going to predict

EPOCHS = 10 
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def Train_model(train_x, train_y, test_x, test_y):
    model = Sequential()
    
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(128, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(2, activation="softmax"))
    
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
    
    filepath = "RNN_Final-{epoch:02d}-{val_acc:.2d}" # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    history = model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(test_x, test_y),
            callbacks=[tensorboard, checkpoint])
    
    #model.save("")
    
    #tensorboard --logdir=logs/
    