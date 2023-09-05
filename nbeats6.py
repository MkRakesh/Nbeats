#!/usr/bin/env python
# coding: utf-8

# In[30]:


import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Create NBeatsBlock custom layer
import tensorflow as tf
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self,
              input_size,
              theta_size,
              horizon,
              n_neurons,
              n_layers,
              **kwargs):
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains 4 fully connected layers
    self.hidden = [tf.keras.layers.Dense(n_neurons , activation = 'relu') for _ in range(n_layers)]
      # Output of block is a theta layer with linear activation
    self.theta_layer =tf.keras.layers.Dense(theta_size, activation = 'linear' ,name = 'theta')

  def call(self ,inputs):
    x = inputs
    for layer in self.hidden:
      x = layer(x)
    theta = self.theta_layer(x)
      #output backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[: ,-self.horizon]
    return backcast, forecast


# In[31]:


# Load the saved model
nbeats_model = tf.keras.models.load_model('NBeats.h5',custom_objects={'NBeatsBlock': NBeatsBlock})

# Customize Streamlit app appearance
# Title

st.text(" Created by:Rakes M K on:05/09/2023 \n Powered by: N-BEATS Algorithm")

st.title("BitPredict")

st.text("Caution: This is not a financial advice !")

input_data_str = st.text_input("Enter last 7 days BitCoin prices separated by spaces:", "")

# Check if the input string is not empty
if input_data_str:
    # Split the input string by spaces and convert to floats
    input_data = [float(value) for value in input_data_str.split()]

    # Check if there are exactly 7 data points
    if len(input_data) == 7:
        # Perform time series forecasting using your model or algorithm
        forecast = nbeats_model.predict([input_data])
        
       
        

        # Display the forecasted value
        st.write(f"Forecast: ${forecast}")
        
        input_data.append(np.squeeze(forecast))
        
        X=[-6,-5,-4,-3,-2,-1,0,1]
        
        forecast=np.squeeze(forecast)
        y = input_data
        df= pd.DataFrame({'X':X ,"y":y})
        plt.style.use('dark_background')
        plt.figure(figsize=(5,2.5))
        plt.plot(df.y[:7])
        plt.plot(df.y[-2:],linestyle='--',label=f'N-BEATS Forecast : ${np.squeeze(forecast)}')
        plt.legend(loc='best',fontsize=9);
        st.pyplot(plt)

    else:
        st.write("Please enter exactly 7 values separated by spaces.")
