# -*- coding: utf-8 -*-

from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60 # use last 60 minute of pricing data

def preprocess_df(df):
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change() # Percentage change between the current and a prior element
            df.dropna(inplace=True) # drop if it creates a "not a number":na
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    sequencial_data = []
    prev_days = deque(maxlen=SEQ_LEN) #prev_days is a deque with a max length of 60...
    #Deques is like a list, you just keep appending to this list, but as list reaches the length of 60  it will pop out the old items and add new ones

    for i in df.values:  #convers dataframe to list
        # i is row of all columns
        prev_days.append([n for n in i[:-1]]) # append all columns except last one (target)
        if len(prev_days) == SEQ_LEN:
            sequencial_data.append([np.array(prev_days), i[-1]]) # we are going to append our features and labels , x and y

    random.shuffle(sequencial_data) # it shuffles the data inside the list
    
    
    #Balancing the data
    buys = []
    sells = []

    for seq, target in sequencial_data:
        if target == 0: # this is a sell, means it goes down
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    #shuffle for good mesure :D
    random.shuffle(buys)
    random.shuffle(sells)

    # we need to find which one is less(we are going to find which one is the minimum)
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower] # if lower is 30k and buys is 70k it will take only 30k of 70k
    sells = sells[:lower] # this aims to balace the data between buys and sells

    sequencial_data = buys + sells
    random.shuffle(sequencial_data)

    X = []
    y = []

    for seq, target in sequencial_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

