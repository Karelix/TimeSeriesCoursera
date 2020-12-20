import numpy as np
import tensorflow as tf

def ex1(lol):
  if lol==1:
    dataset = tf.data.Dataset.range(10)
    for val in dataset:
      print(val.numpy())
  elif lol==2:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1)
    for window_dataset in dataset:
      for val in window_dataset:
        print(val.numpy(),end=' ')
      print()
  elif lol==3:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1,drop_remainder=True)
    for window_dataset in dataset:
      for val in window_dataset:
        print(val.numpy(),end=' ')
      print()
  elif lol==4:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1,drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    for window in dataset:
      print(window.numpy())
  elif lol==5:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1,drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1],window[-1:]))
    for x,y in dataset:
      print(x.numpy(),y.numpy())
  elif lol==6:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1,drop_remainder=True)
    dataset =  dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1],window[-1:]))
    dataset = dataset.shuffle(buffer_size=10)
    for x,y in dataset:
      print(x.numpy(),y.numpy())
  elif lol==7:
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5,shift=1,drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1],window[-1:]))
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(2).prefetch(1)
    for x,y in dataset:
      print('x =',x.numpy())
      print('y =',y.numpy())

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1,shift=1,drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1],window[-1:]))
  # dataset = dataset.shuffle(shuffle_buffer)
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

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