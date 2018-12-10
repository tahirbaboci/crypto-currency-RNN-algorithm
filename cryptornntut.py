import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random



#Close : is the price of the end of 60 seconds interval

SEQ_LEN = 60 # use last 60 minute of pricing data
FUTURE_PERIOD_PREDICT = 3 # next minutes
RATIO_TO_PREDICT = "LTC-USD" # which one we are going to predict

main_df = pd.DataFrame() # merging data frames # this is just an empty data frame



def classify(current, future):
    if float(future) > float(current):
        return 1 # means is good thing and you should buy this
    else:
        return 0 # means is not good thing and you shouldn't buy this




def preprocess_df(df):
    df = test_main_df
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change() # ?
            df.dropna(inplace=True) # drop if it creates a "not a number":na
            df[col] = preprocessing.scale(df[col].values)


    df.dropna(inplace=True) # drop if it creates a "not a number":na
    sequencial_data = []
    prev_days = deque(maxlen=SEQ_LEN) #prev_days is a deque with a max length of 60...
    #Deques is like a list, you just keep appending to this list, but as list reaches the length of 60  it will pop out the old items and add new ones

    for i in df.values:  #convers dataframe to list of lists
        # i is row of all columns
        prev_days.append([n for n in i[:-1]]) # append all columns except last one (target)
        if len(prev_days) == SEQ_LEN:
            sequencial_data.append([np.array(prev_days), i[:-1]]) # we are going to append our features and labels , x and y

    random.shuffle(sequencial_data) # ?? why
    #print(sequencial_data)
    buys = {}
    sells = {}

    for seq, target in sequencial_data:
        if target == 0: # this is a sell, means it goes down
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    #shuffle for good mesure :D
    random.shuffle(buys)
    random.shuffle(sells)

    # we need to find which one is lesser(we are going to find which one is the minimum)
    lower = min(len(buys), len(sells))
    # buys up too lower
    buys = buys[:lower] # ?
    #sells up too lower
    sells = sells[:lower] # ?

    sequencial_data = buys + sells
    random.shuffle(sequencial_data)

    X = []
    y = []

    for seq, target in sequencial_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


#pd.show_versions()

ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD'] # files that are going to be used
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])
    #print(df.head())
    df.rename(columns={"close" : f"{ratio}_close", "volume" : f"{ratio}_volume"}, inplace=True) #inplace is in case so we dont need to redifine dataframe

    
    df.set_index("time", inplace=True)
    #df = df.drop('low', 1) #0 means index 1 means column
    #df = df.drop('high', 1)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume
    
    print(df.head())

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

#print(main_df.head())

#for c in main_df.columns:
#    print(c)





main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) # it will check 3 periods of the data in the future
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"])) #map close and future parameters to classify function
print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(15))


#time           close           future     target
#124124124   96.38999999     96.47000111      1    # Here!!!!
#234234234   96.51232323     96.40000211      0
#452342342   96.44000233     96.44000233      0
#623423454   96.47000111     96.40000002      0    # we can see 96.47000111






# we have to build the sequences
# we have to balace the data
# we have to normalize the data
# we have to scale the data


# from now we are ready to make some sequenceses and train the model

times = sorted(main_df.index.values)
last_5percent = times[-int(0.05*len(times))] # get the time of last 5 percent === -int(0.05*len(times))
print(last_5percent)

test_main_df = main_df[(main_df.index >= last_5percent)] # it takes last 5% of the data
main_df = main_df[(main_df.index < last_5percent)]  # it takes the rest (first 95%)


train_x, train_y = preprocess_df(main_df)
test_x, test_y = preprocess_df(test_main_df)


print(f"train data: {len(train_x)} validation: {len(test_x)}")
print(f"Dont buys: {train_y.count(0)}. buys: {train_y.count(1)}")
print(f"TEST Dont buys: {test_y.count(0)}, buys: {test_y.count(1)}")
