import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def ex1(lol):
  if lol==1:
    model = keras.models.Sequential([
      keras.layers.SimpleRNN(20,return_sequences=True,input_shape=[None,1]), # None is the sequence length we set it to none so that RNN can handle sequences of any length
      keras.layers.SimpleRNN(20),
      keras.layers.Dense(1)
    ])
  elif lol==2:
    model = keras.models.Sequential([
      keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]), # only none because now we take sequences of any length
      keras.layers.SimpleRNN(20,return_sequences=True),
      keras.layers.SimpleRNN(20),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x:x*100.0)
    ])
  elif lol==3:
    train_set = windowed_dataset(x_train,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)
    model = keras.models.Sequential([
      keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]),
      keras.layers.SimpleRNN(40,return_sequences=True),
      keras.layers.SimpleRNN(40),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x:x*100.0)
    ])

    lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))
    optimizer = keras.optimizers.SGD(lr=1e-8,momentum=0.9)
    model.compile(loss=keras.losses.Huber(),optimizer=optimizer,metrics=['mae'])
    history = model.fit(train_set,epochs=100,callbacks=[lr_schedule])
  elif lol==4:
    train_set = windowed_dataset(x_train,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)
    model = keras.models.Sequential([
      keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]),
      keras.layers.SimpleRNN(40,return_sequences=True),
      keras.layers.SimpleRNN(40),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x:x*100.0)
    ])
    
    optimizer = keras.optimizers.SGD(lr=5e-5,momentum=0.9)
    model.compile(loss=keras.losses.Huber(),optimizer=optimizer,metrics=['mae'])
    model.fit(train_set,epochs=500)
  elif lol==5:
    train_set = windowed_dataset(x_train,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)
    model = keras.models.Sequential([
      keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]),
      keras.layers.Bidirectional(keras.layers.LSTM(32,return_sequences=True)),
      keras.layers.Bidirectional(keras.layers.LSTM((32)),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x:x*100.0)
    ])
    
    optimizer = keras.optimizers.SGD(lr=1e-6,momentum=0.9)
    model.compile(loss=keras.losses.Huber(),optimizer=optimizer,metrics=['mae'])
    model.fit(train_set,epochs=100)
  elif lol==6:
    train_set = windowed_dataset(x_train,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)
    model = keras.models.Sequential([
      keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]),
      keras.layers.Bidirectional(keras.layers.LSTM(32,return_sequences=True)),
      keras.layers.Bidirectional(keras.layers.LSTM(32,return_sequences=True)),
      keras.layers.Bidirectional(keras.layers.LSTM((32)),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x:x*100.0)
    ])
    
    optimizer = keras.optimizers.SGD(lr=1e-6,momentum=0.9)
    model.compile(loss=keras.losses.Huber(),optimizer=optimizer,metrics=['mae'])
    model.fit(train_set,epochs=100)
  elif lol==7:

def single_layer_nn(series):
  window_size = 20
  batch_size = 32
  shuffle_buffer_size = 1000
  dataset = windowed_dataset(series,window_size,batch_size,shuffle_buffer_size)
  l0 = tf.keras.layers.Dense(1,input_shape=[window_size]) # we want to print the leant weights
  model = tf.keras.Sequential([l0])
  model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD(lr=1e-6,momentum=0.9))
  model.fit(dataset,epochs=100,verbose=0)
  print('Layer weights {}'.format(l0.get_weights()))
  print(series[1:21])
  model.predict(series[1:21][np.newaxis])
  forecast = []
  for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))
  forecast = forecast[split_time-window_size:]
  results = np.array(forecast)[:,0,0]
# def ex2(lol):
#   if 

if __name__ == '__main__':
  ex1(3)