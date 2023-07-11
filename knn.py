import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import ta
import yfinance as yf
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

 
class Backtest:
    def __init__(self, symbol='ACC.NS', tim='1y', sd=9):
        self.tim = tim
        self.symbol = symbol
        self.df = yf.download(tickers=symbol, period = self.tim)
        self.src = self.df["Close"].values
        self.h = sd
        y2, y1 = self.nadaraya_watson_envelope()
        self.gen_signals(y1,y2)


    def nadaraya_watson_envelope(self):
        n = len(self.src)
        y2 = np.empty(n)
        y1 = np.empty(n)
        h= self.h
        for i in range(n):
            sum = 0
            sumw = 0
            for j in range(n):
                w = np.exp(-(np.power(i-j,2)/(h*h*2)))
                sum += self.src[j]*w
                sumw += w
            y2[i] = sum/sumw
            if i > 0:
                y1[i] = (y2[i] + y2[i-1]) / 2
        self.df['y2'] = y2
        self.df['y1'] = y1
        return y2, y1
    
    def gen_signals(self,y1,y2):
        buy_signals = []
        sell_signals = []
        thld = 0.01

        for i in range(1, len(y2)):
            d = y2[i] - y2[i-1]
            if d > thld and y2[i-1] < y1[i-1]:
                buy_signals.append(i)
            elif d < -thld and y2[i-1] > y1[i-1]:
                sell_signals.append(i)
        money = 100
        trades = 0
        profit = []
        for i in range(len(buy_signals)):
            buy_index = buy_signals[i]
            if i < len(sell_signals):
                sell_index = sell_signals[i]
                trades += 1
                money *= self.src[sell_index] / self.src[buy_index]
                profit.append(money - 100)
        self.profit  = pd.DataFrame(profit)
        self.rets = "Returns "+ self.tim +" = " + str(round(((money/100-1)*100),2)) + "%"
        self.trades = "Total Trades: " + str(trades)
        self.roi = "Total Return: " + str(round((money-100),2)) + "%"
        self.avg_return = "Average Return Per Trade: " + str(round((money-100)/trades,2)) + "%"
        self.win_rate = "Win Rate: " + str(round((len([x for x in profit if x > 0])/trades)*100,2)) + "%"
        plt.figure(figsize=(30,10))
        plt.plot(y2,color='blue')
        plt.plot(self.src,color='black', label='close')
        for signal in buy_signals:
            plt.axvline(x=signal, color='green',linewidth=2)

        for signal in sell_signals:
            plt.axvline(x=signal, color='red',linewidth=2)
        plt.legend()
        plt.show()
ticks = pd.read_csv("100_tick.csv")
clck = ticks['Symbol'].values
selected_option = st.selectbox("Select an Stock",clck)
st.write("You selected: ", selected_option)
tim = "1y" #st.text_input("time frame ex: 1y,2y,100d,10d..")
std = 8 #st.text_input("Standard devation, like 5,6,7,8,9....")
backtest = Backtest(selected_option, tim, int(std))
st.pyplot(plt)
profits = backtest.profit.iloc[:, 0]
pro = backtest.profit # Remove extra column

metrics = {
    "Total Trades": len(profits),
    "Total Return": str((round(profits.iloc[-1],2))-30)+"%",
    "Risk appetite": str((round(profits.mean(), 2)-30))+"%",
    "Std Dev of Daily Returns": round(pro.iloc[:, 0].std(), 2),
    "Sharpe Ratio": round((backtest.profit.iloc[:, 0].mean()) / backtest.profit.iloc[:, 0].std(), 2),
    "Max Drawdown": round(profits.min(), 2),
    "Winning Trades": round(len(profits[profits > 0]), 2),
    "Profit Factor": round(abs(profits[profits > 0].sum() / 2), 2)
}
df = pd.DataFrame.from_dict(metrics,orient="index", columns=["Value"])
print(df)
st.write("Backtest on 1 year")
st.table(df)


class methode:
    @staticmethod
    def predict_next_close(sym):
        try:
            data = yf.download(sym, period='3y')
            
            data['Close_next'] = data['Close'].shift(-1)
            data['Close_pct_change'] = data['Close_next'].pct_change() *100
            data = data.fillna(0)
            
            # Calculate technical indicators
            data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['stoch'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
            data['stochrsi'] = ta.momentum.StochRSIIndicator(data['Close']).stochrsi()
            data['williams'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
            data['roc'] = ta.momentum.ROCIndicator(data['Close']).roc()
            data['tsi'] = ta.momentum.TSIIndicator(data['Close']).tsi()
            data['uo'] = ta.momentum.UltimateOscillator(data['High'], data['Low'], data['Close']).ultimate_oscillator()
            data['kama'] = ta.momentum.KAMAIndicator(data['Close']).kama()
            data['mfi'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
            data['adx'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
            data['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
            data['dpo'] = ta.trend.DPOIndicator(data['Close']).dpo()
            data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            data = data.dropna()
            index = pd.DataFrame(data.columns)
            index = index[0].to_numpy()[8:]
            
            X = data[index].values
            y = data[['Close_next', 'Close_pct_change']].values
            
            # # Train the KNN model
            model = KNeighborsRegressor()
            model.fit(X, y)
            # Prepare feature data for the next day
            last_day_data = data.iloc[-1][index].values.reshape(1, -1)  # Use the last day's technical indicator values
            
            # Predict the next closing price and percentage change
            next_close, next_pct_change = model.predict(last_day_data)[0]
            
            # Get the last row of the DataFrame
            last_row = data.tail(1)
            
            # Predict the next closing price and percentage change
            next_close, next_pct_change = model.predict(last_day_data)[0]
            
            # Add the predicted next closing price and percentage change to the DataFrame
            last_row['next_close'] = next_close
            last_row['next_pct_change'] = next_pct_change
            
            # Add the stock name to the DataFrame
            last_row['stock_name'] = sym


            return last_row

        except Exception as e:
            print(f"Error occurred: {e}")
            return None


class prediction:

    def knn_100():
        result = pd.DataFrame()
        ok = pd.read_csv("100_tick.csv")
        symbols = ok['Symbol']
        for sym in symbols:
            prediction = methode.predict_next_close(sym)
            row = pd.DataFrame(prediction)
            result = pd.concat([result, row], ignore_index=True)

        return result
            
class metrics:

    def uptrend(df):
        return df[df['next_pct_change'] > 1]['stock_name'].tolist()
    

    def downtrend(df):
        return df[df['next_pct_change'] < -1]['stock_name'].tolist()

    def golden_crossover(df):
        url='https://www.screener.in/screens/336509/golden-crossover/'
        gc=pd.read_html(url)[0][['Name','50 DMA  Rs.','200 DMA  Rs.','P/E']].dropna()
        return gc[:-5]

    # todo:1 stocks in reversal zones (expected uptrend)
    def up_reversal(df):
        rev = df[(df['rsi'] < 40) & (df['stoch'] < 30) & (df['next_pct_change'] > 1)][['rsi','stoch','next_pct_change','stock_name']]
        return rev
    
    # todo:2 stocks in reversal zone (expected downtrend)
    def down_reversal(df):
        rev = df[(df['rsi'] > 60) & (df['stoch'] > 80) & (df['next_pct_change'] < -1)][['rsi','stoch','next_pct_change','stock_name']]
        return rev

    # todo:3 give list of predicted closing price on current day
    def list_pred(df):
        return df[['next_pct_change','stock_name']]


