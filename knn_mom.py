import warnings
warnings.filterwarnings('ignore')
import ta
import yfinance as yf
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


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
    
ok = prediction.pred.knn_100()
print(metrics.uptrend(ok))
print(metrics.downtrend(ok))
print(metrics.golden_crossover(ok))
print(metrics.up_reversal(ok))
print(metrics.down_reversal(ok))