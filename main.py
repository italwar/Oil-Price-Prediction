import streamlit as st
from datetime import date
from fbprophet import Prophet
from tensorflow.keras.models import load_model
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle


st.set_option('deprecation.showPyplotGlobalUse', False)
model = load_model('model/')
@st.cache
def main():
	

	with open("FBMODEL.pkl", 'rb') as f:
		m = pickle.load(f)

	df = pd.read_csv(r"C:\Users\hp\Downloads\BrentOilPrices.csv")

	df["Date"] = pd.to_datetime(df["Date"])

	df.set_index("Date",inplace=True)

	return df,m

df,m = main()

menu = ["Brent oil","LSTM","Fb Prophet"]

choice = st.sidebar.selectbox("Menu",menu)


@st.cache
def df_x_y(df,window_size = 5):
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i+5]
        x.append(row)
        label = df_as_np[i+5]
        y.append(label)
    return np.array(x),np.array(y)
        
@st.cache
def new_data(df,x):
    for i in range(x):
        test = df[["predicted"]].tail(5)
        test.reset_index(drop = True,inplace=True)
        test_as_np = test.to_numpy()
        row = test_as_np[-5:]
        x_t = []
        x_t.append(row)
        x_t = np.array(x_t)
        pred = model.predict(x_t).flatten()
        ff = pd.DataFrame({"predicted" : pred})
        
        #x_t.append(row)
        df = df.append(ff).reset_index(drop=True)
    return df[["predicted"]]
    
    


x,y = df_x_y(df)
#st.write(len(x))
y = np.reshape(y,(-1,1))

y = np.array(y)

x_train,y_train = x[:len(df)], y[:len(df)]


if choice == "Brent oil":



	st.header("Brent Crude Oil")

	st.write("Brent Crude oil is a major benchmark price for purchases of oil worldwide. While Brent Crude oil is sourced from the North Sea the oil production coming from Europe, Africa and the Middle East flowing West tends to be priced relative to this oil. The Brent prices displayed in Trading Economics are based on over-the-counter (OTC) and contract for difference (CFD) financial instruments. Our market prices are intended to provide you with a reference only, rather than as a basis for making trading decisions. Trading Economics does not verify any data and disclaims any obligation to do so.")

	st.subheader("Visualization Of Crude Oil 1988 - 2021")

	plt.figure(figsize=(16,4))
	plt.plot(df)
	st.pyplot()


elif choice == "LSTM":
	#cc = new_data(train_p,300)

	st.header("FORECASTE USING LSTM MODEL")

	T = st.selectbox("Select the range of Forecasting",["One day" , "One Week" , "One Month","Six Month"])
	if T == "One day":
		T = 1
	elif T == "One Week":
		T = 7
	elif T == "One Month":
		T = 30
	elif T == "Six Month":
		T = 180

	tp = model.predict(x_train).flatten()
	train_p = pd.DataFrame({'actual' : y_train.flatten() , 'predicted' : tp})
	#st.write(train_p)



	bt=st.button("Forecast")
	if bt is True:
		cc = new_data(train_p,T)
		res = cc.tail(T)
		st.write(res)
		plt.figure(figsize=(16,4))
		plt.plot(res)
		st.pyplot()

	#st.write("***ON PROGRESS***")

     #datelist = pd.date_range(datetime.today(), periods=100).tolist()

elif choice == "Fb Prophet":

	st.header("FORECASTE USING FB PROPHET")

	T = st.selectbox("Select the range of Forecasting",["One day" , "One Week" , "One Month","Six Month"])
	if T == "One day":
		T = 1
	elif T == "One Week":
		T = 7
	elif T == "One Month":
		T = 30
	elif T == "Six Month":
		T = 180

	bt2 = st.button("Forecast")

	if bt2 is True:


	    predict = m.make_future_dataframe(periods=T,freq="D")
	    forcast = m.predict(predict)
	    #st.write(forcast[["ds","yhat"]].tail(T))
	    res = forcast[["ds","yhat"]].tail(T)
	    res.columns = ["Date" , "Price"]
	    res.set_index("Date",inplace=True)
	    st.write(res)

	    plt.figure(figsize=(16,4))
	    plt.plot(res)
	    st.pyplot()

