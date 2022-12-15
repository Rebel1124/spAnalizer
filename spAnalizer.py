 ## UTOR: FinTech Bootcamp - Project 2: Trading Strategy Optimizer


if __name__ == '__main__':   
    

    # Import libraries
    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import DateOffset
    pd.set_option('mode.chained_assignment', None)
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import time
    import datetime
    from datetime import date, timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit as st
    from PIL import Image

    # Pandas Technical Analysis Library
    # https://github.com/twopirllc/pandas-ta
    import pandas_ta as ta    
    
    # sklearn Random Forest Classifier and metrics libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
       
    
    # Import S&P Index Dataframe
    
    sp = pd.read_csv('spIndex.csv')
    symbols = sp['Symbol'].tolist()
    names = sp['Description'].tolist()
    
    
    # Standard/Set Inputs

    # Time Interval
    timePeriod="max"
    daily="1d"
    now = datetime.date.today()

    # Portfolio Metrics
    initial = 10000
    riskFree = 0.01
    #risk=0.1

    # Pct_Change - days ahead
    days = 1



    # Streamlit Setup
    # STreamlit Asset Inputs
    
    #st.markdown(
    #    """
    #    <style>
    #    .main {
    #    background-color: #F5F5F5;
    #    }
    #    </style>
    #    """,
    #    unsafe_allow_html=True
    #)
   
    image = Image.open('arrow.png')    

    st.image(image, width=100)

    st.markdown("<h1 style='text-align: left; color: Purple; padding-left: 0px; font-size: 60px'>ARROW-UP CAPITAL</h1>", unsafe_allow_html=True)



    st.markdown("<h2 style='text-align: left; color: teal; padding-left: 0px; font-size: 50px'>Trading Strategy Optimizer</h2>", unsafe_allow_html=True)

    # User to choose asset
    colc, cold, cole = st.columns([2, 3, 7])

    #assetCodesList = ['CL=F', 'GC=F', '^RUT', '^GSPC', 'EURUSD=X', 'GBPJPY=X', 'BTC-USD', 'ETH-USD']
    #assetNamesList = ['crudeOil', 'Gold', 'Russel2000', 'S&P500', 'EUR-USD', 'GBP-JPY', 'BTC-USD', 'ETH-USD']
    
    assetCodesList = symbols
    assetNamesList = names
    

    colc.markdown('')
    colc.markdown('')
    colc.markdown("<h3 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Asset Selection<b></h3>", unsafe_allow_html=True)
    assetNames = cold.selectbox('', assetNamesList, index=0)
    index = assetNamesList.index(assetNames)
    assetCodes = assetCodesList[index]
    
    
    if(assetNames=='BTC-USD'):
        minValue = 36
        maxValue = 84
        val = 70
    elif(assetNames=='ETH-USD'):
        minValue = 12
        maxValue = 60
        val = 42
    else:
        minValue = 12
        maxValue = 1260
        val = 145
        

    # streamlit sidebar and strategy indicator settings setup

    chessImage = Image.open('chess.png')
    st.sidebar.image(chessImage, use_column_width=True)
    st.sidebar.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Trading Strategy Inputs<b></h3>", unsafe_allow_html=True)
    st.sidebar.markdown(' ')
    
    # Risk Limit (Strategy Stop Loss)
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Risk Limit<b></h4>", unsafe_allow_html=True)
    risk = (st.sidebar.number_input('Position Stop/Loss (%)', min_value=0, max_value=100, step=5, value=10))/100
    
    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    
    # Date Settings
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Date Settings<b></h4>", unsafe_allow_html=True)
    start_date = st.sidebar.date_input('Start Date',date(2006,6,30)).strftime("%Y-%m-%d")
    end_date = st.sidebar.date_input('End Date',now).strftime("%Y-%m-%d")
    training_months = st.sidebar.number_input('Training Months', min_value=minValue, max_value=maxValue, step=1, value=val)
    
    
    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    
    # Feature Settings
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Feature Settings<b></h4>", unsafe_allow_html=True)
    rollingWindow = st.sidebar.number_input('zScores - rolling window', min_value=24, max_value=48, step=1, value=30)
    deltaDays = st.sidebar.number_input('normalization - delta days', min_value=1, max_value=5, step=1, value=1)


    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    
    # Random Forest Model Settings
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Random Forest Model<b></h4>", unsafe_allow_html=True)
    estimators = st.sidebar.number_input('Estimators', min_value=50, max_value=5000, step=1, value=500)
    depth = st.sidebar.number_input('Max Depth', min_value=2, max_value=100, step=1, value=50)
    samples_split = st.sidebar.number_input('Min Samples Split', min_value=1, max_value=500, step=1, value=2)
    samples_leaf = st.sidebar.number_input('Min Samples Leaf', min_value=1, max_value=500, step=1, value=1)


    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')

    # Daily Indicators
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Daily Indicator Settings<b></h4>", unsafe_allow_html=True)
    dailyEMAShort = st.sidebar.number_input('EMA Short', min_value=9, max_value=18, step=1, value=12)
    dailyEMALong = st.sidebar.number_input('EMA Long', min_value=18, max_value=32, step=1, value=26)
    smaLength = st.sidebar.number_input('Simple Moving Average', min_value=1, max_value=200, step=1, value=50)
    rsiLength = st.sidebar.number_input('Relative Strength Index', min_value=6, max_value=30, step=1, value=14)
    overbought = st.sidebar.number_input('RSI - Overbought', min_value=75, max_value=85, step=1, value=80)
    oversold = st.sidebar.number_input('RSI - Oversold', min_value=15, max_value=25, step=1, value=20)
    neutral = st.sidebar.number_input('RSI - Neutral', min_value=45, max_value=55, step=1, value=50)
    rocLength = st.sidebar.number_input('Rate of Change Index', min_value=6, max_value=30, step=1, value=21)
    momLength = st.sidebar.number_input('Momentum Index', min_value=6, max_value=30, step=1, value=14)
    bbandsLength = st.sidebar.number_input('Bollinger Bands', min_value=24, max_value=48, step=1, value=30)
    macdFast = st.sidebar.number_input('MACD Fast', min_value=6, max_value=18, step=1, value=12)
    macdSlow = st.sidebar.number_input('MACD Slow', min_value=18, max_value=32, step=1, value=26)     
    macdSignal = st.sidebar.number_input('MACD Signal', min_value=6, max_value=21, step=1, value=9)

    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')

    # Indicator Names

    # Number of Assets
    count = len(assetCodes)

    # Daily Indicators
    dailyEMAShortIndicatorName = 'EMA_'+str(dailyEMAShort)
    newDailylyEMAShortIndicatorName = 'EMAShort'

    dailyEMALongIndicatorName = 'EMA_'+str(dailyEMALong)
    newDailyEMALongIndicatorName = 'EMALong'

    rsiIndicatorName = 'RSI_'+str(rsiLength)
    newRsiIndicatorName = 'RSIline'

    momIndicatorName = 'MOM_'+str(momLength)
    newMomIndicatorName = 'MOMline'

    rocIndicatorName = 'ROC_'+str(rocLength)
    newRocIndicatorName = 'ROCline'

    smaIndicatorName = 'SMA_'+str(smaLength)
    newSmaIndicatorName = 'SMAline'

    bollingerLowerIndicatorName = 'BBL_'+str(bbandsLength)+'_2.0'
    newBollingerLowerIndicatorName = 'lowerBB'

    bollingerMiddleIndicatorName = 'BBM_'+str(bbandsLength)+'_2.0'
    newBollingerMiddleIndicatorName = 'middleBB'

    bollingerUpperIndicatorName = 'BBU_'+str(bbandsLength)+'_2.0'
    newBollingerUpperIndicatorName = 'upperBB'

    bollingerStdIndicatorName = 'BBB_'+str(bbandsLength)+'_2.0'
    newBollingerStdIndicatorName = '2stdBB'

    macdIndicatorName = "MACD_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDIndicatorName = "MACDline"

    macdHistogramIndicatorName = "MACDh_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDHistogramIndicatorName = "MACDHistogram"

    macdSignalIndicatorName = "MACDs_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDSignalIndicatorName = "MACDSignal"

    featuresRaw = [newDailylyEMAShortIndicatorName, newDailyEMALongIndicatorName, newRsiIndicatorName, newMomIndicatorName, newRocIndicatorName, 
                newSmaIndicatorName, newBollingerLowerIndicatorName, newBollingerMiddleIndicatorName, newBollingerUpperIndicatorName, 
                newBollingerStdIndicatorName, newMACDIndicatorName, newMACDHistogramIndicatorName, newMACDSignalIndicatorName]


    # Number of RawFeatures
    featuresCount = len(featuresRaw)
    
    @st.cache(allow_output_mutation=True)
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    # Function to calculate daily Indicators
    @st.cache(allow_output_mutation=True)
    def dailyIndicators(assetCode, assetName, timePeriod, daily, days, start_date, end_date, dailyEMAShort, dailyEMALong, rsiLength, momLength, rocLength, 
                        smaLength, bbandsLength, macdFast, macdSlow, macdSignal, dailyEMAShortIndicatorName, dailyEMALongIndicatorName,
                        rsiIndicatorName, momIndicatorName, rocIndicatorName, smaIndicatorName, bollingerLowerIndicatorName, bollingerMiddleIndicatorName, bollingerUpperIndicatorName, bollingerStdIndicatorName, macdIndicatorName,
                        macdHistogramIndicatorName, macdSignalIndicatorName, newDailylyEMAShortIndicatorName,
                        newDailyEMALongIndicatorName, newRsiIndicatorName, newMomIndicatorName, newRocIndicatorName,newSmaIndicatorName,
                        newBollingerLowerIndicatorName, newBollingerMiddleIndicatorName, newBollingerUpperIndicatorName,
                        newBollingerStdIndicatorName, newMACDIndicatorName, newMACDHistogramIndicatorName,
                        newMACDSignalIndicatorName):
        
        # Get Daily Asset Data

        dfDaily = pd.DataFrame()
        dfDaily = dfDaily.ta.ticker(assetCode, period=timePeriod, interval=daily)
        dfDaily = dfDaily[(dfDaily.index > start_date) & (dfDaily.index < end_date)]
        dfDaily = dfDaily[(dfDaily.Close > 0)]
        
        # Create Ticker Column
        dfDaily["Ticker"] = assetName
        dfDaily["Date"] = dfDaily.index

        # Use the pct_change function to generate returns from close prices
        dfDaily["ActualReturns"] = dfDaily["Close"].pct_change(days).shift(-days)
        
        # Drop all NaN values from the DataFrame
        dfDaily = dfDaily.dropna()

        # Initialize the new Signal column
        dfDaily['Signal'] = 0.0

        # When Actual Returns are greater than or equal to 0, generate signal to buy asset long
        dfDaily.loc[(dfDaily['ActualReturns'] >= 0), 'Signal'] = 1

        # When Actual Returns are less than 0, generate signal to sell asset short
        dfDaily.loc[(dfDaily['ActualReturns'] < 0), 'Signal'] = -1


        # Create your own Custom Strategy
        CustomStrategyDaily = ta.Strategy(
            name="Daily Indicators",
            description="daily Trading Indicators",
            ta=[
                {"kind": "ema", "length": dailyEMAShort},
                {"kind": "ema", "length": dailyEMALong},
                {"kind": "rsi", "length": rsiLength},
                {"kind": "mom", "length": momLength},
                {"kind": "roc", "length": rocLength},
                {"kind": "sma", "length": smaLength},
                {"kind": "bbands", "length": bbandsLength},
                {"kind": "macd", "fast": macdFast, "slow": macdSlow, "signal": macdSignal},
            ]
        )


        # Run "Custom Daily Strategy"
        dfDaily.ta.strategy(CustomStrategyDaily)
        dfDaily=dfDaily.dropna()
        algoDataDaily = dfDaily[['Ticker', 'Date', 'Close', 'ActualReturns','Signal',dailyEMAShortIndicatorName, dailyEMALongIndicatorName,
                            rsiIndicatorName, momIndicatorName, rocIndicatorName, smaIndicatorName, bollingerLowerIndicatorName,
                            bollingerMiddleIndicatorName, bollingerUpperIndicatorName, bollingerStdIndicatorName,
                            macdIndicatorName, macdHistogramIndicatorName, macdSignalIndicatorName]]

        algoDataDaily = algoDataDaily.rename({dailyEMAShortIndicatorName: newDailylyEMAShortIndicatorName,
                                dailyEMALongIndicatorName: newDailyEMALongIndicatorName,
                                rsiIndicatorName: newRsiIndicatorName,
                                momIndicatorName: newMomIndicatorName,
                                rocIndicatorName: newRocIndicatorName,
                                smaIndicatorName: newSmaIndicatorName,
                                bollingerLowerIndicatorName: newBollingerLowerIndicatorName,
                                bollingerMiddleIndicatorName: newBollingerMiddleIndicatorName,
                                bollingerUpperIndicatorName: newBollingerUpperIndicatorName,
                                bollingerStdIndicatorName: newBollingerStdIndicatorName,
                                macdIndicatorName: newMACDIndicatorName,
                                macdHistogramIndicatorName: newMACDHistogramIndicatorName,
                                macdSignalIndicatorName: newMACDSignalIndicatorName}, axis=1)
        
        return algoDataDaily
    
    # Calculate Feature Z_Scores
    @st.cache(allow_output_mutation=True)
    def Zscore(algoData, featuresRaw, rollingWindow):

        for i in featuresRaw:
            algoData[i+'_zscore'] = (algoData[i] - algoData[i].rolling(window=rollingWindow).mean())/algoData[i].rolling(window=rollingWindow).std()
            algoData = algoData.ffill(axis = 0)
            
        return algoData



    # Normalize Features with Price
    @st.cache(allow_output_mutation=True)
    def Normalize(algoData, featuresRaw, deltaDays):

        for i in featuresRaw:
            
            if((i=='EMAShort') | (i=='EMALong') | (i=='SMAline')):
            
                algoData[i+'_normal'] = (algoData[i].diff(periods=deltaDays))
                
            elif((i=='lowerBB') | (i=='middleBB') | (i=='upperBB')):
                
                algoData[i+'_normal'] = (algoData['Close']/algoData[i])
                
            elif ((i=='RSIline') | (i=='MOMline') | (i=='ROCline') ):
                
                algoData[i+'_normal'] = (algoData[i]/algoData['2stdBB'])
                
            elif ((i=='MACDline') | (i=='MACDSignal') | (i=='MACDHistogram')):
                
                algoData[i+'_normal'] = (algoData[i])
            
            elif((i == '2stdBB')):
                
                algoData[i+'_normal'] = (algoData[i]/algoData['Close'])
            
            else:
                algoData[i+'_normal'] = 0
            
            
        algoData = algoData.ffill(axis = 0)
        
        return algoData
    
    # Calculate percentage Asset Returns
    @st.cache(allow_output_mutation=True)
    def pctReturns(algoData, assetNames):
        
        lastDay = algoData['Close'][-1]
        oneDay = algoData['Close'][-2]
        oneWeek = algoData['Close'][-6]
        oneMonth = algoData['Close'][-22]
        oneQuarter = algoData['Close'][-65]
        oneYear = algoData['Close'][-365]
        
        oneDaypct = round((lastDay/oneDay)-1,6)
        oneWeekpct = round((lastDay/oneWeek)-1,6)
        oneMonthpct = round((lastDay/oneMonth)-1,6)
        oneQuarterpct = round((lastDay/oneQuarter)-1,6)
        oneYearpct = round((lastDay/oneYear)-1,6)
        
        
        colours = ['lavender']
        val = [assetNames]

        pctReturns = [oneDaypct, oneWeekpct, oneMonthpct, oneQuarterpct, oneYearpct]
        
        
        for ret in pctReturns:
            if(ret > 0):
                colours.append('#98FB98')
                val.append('{:.2%}'.format(ret)+'  ↑')
            elif(ret < 0):
                colours.append('#F08080')
                val.append('{:.2%}'.format(ret)+'  ↓')
            else:
                colours.append('#E3CF57')
                val.append('{:.2%}'.format(ret)+'  -')

        
        head = ['Returns', 'Day', 'Week', 'Month', 'Quarter', 'Year']


        fig14 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='#551A8B',
                    align='center',
                    font=dict(color='white')),
            cells=dict(values=val,
                fill_color=colours,
                align='center'))
        ])

        fig14.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=425, height=50)

        return fig14
    
    
    # Calculate algo Signals
    @st.cache(allow_output_mutation=True)
    def algoSignals(allStrategyReturnsTest, assetNames):
        
        
        macd = allStrategyReturnsTest['MACDStrategy'][-1]
        rsi = allStrategyReturnsTest['RSIStrategy'][-1]
        impulse = allStrategyReturnsTest['ImpulseStrategy'][-1]
        bollinger = allStrategyReturnsTest['BBStrategy'][-1]
        
        
        colours = ['lavender']
        val = [assetNames]

        algoStrategy = [macd, rsi, impulse, bollinger]
        
        
        for ret in algoStrategy:
            if(ret > 0):
                colours.append('#98FB98')
                val.append('Buy')
            elif(ret < 0):
                colours.append('#F08080')
                val.append('Sell')
            else:
                colours.append('#E3CF57')
                val.append('Hold')

        
        head = ['Algo', 'MACD', 'RSI', 'Impulse', 'Bollinger']


        fig15 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='#551A8B',
                    align='center',
                    font=dict(color='white')),
            cells=dict(values=val,
                fill_color=colours,
                align='center'))
        ])

        fig15.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=425, height=50)

        return fig15
    
    
    
    
    # Calculate algo Signals
    @st.cache(allow_output_mutation=True)
    def rfSignals(allStrategyReturnsTest, assetNames):
        
        
        standard = allStrategyReturnsTest['pred_standard'][-1]
        zscores = allStrategyReturnsTest['pred_zscores'][-1]
        normalized = allStrategyReturnsTest['pred_normalized'][-1]
        
        
        colours = ['lavender']
        val = [assetNames]

        rfStrategy = [standard, zscores, normalized]
        
        
        for ret in rfStrategy:
            if(ret > 0):
                colours.append('#98FB98')
                val.append('Buy')
            elif(ret < 0):
                colours.append('#F08080')
                val.append('Sell')
            else:
                colours.append('#E3CF57')
                val.append('Hold')

        
        head = ['RF', 'standard', 'zScores', 'normalized']


        fig16 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='#551A8B',
                    align='center',
                    font=dict(color='white')),
            cells=dict(values=val,
                fill_color=colours,
                align='center'))
        ])

        fig16.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=425, height=50)

        return fig16
    
    
    
    algoData = dailyIndicators(assetCodes, assetNames, timePeriod, daily, days, start_date, end_date, dailyEMAShort, dailyEMALong, rsiLength, momLength, rocLength, 
                    smaLength, bbandsLength, macdFast, macdSlow, macdSignal, dailyEMAShortIndicatorName, dailyEMALongIndicatorName,
                    rsiIndicatorName, momIndicatorName, rocIndicatorName, smaIndicatorName, bollingerLowerIndicatorName, bollingerMiddleIndicatorName, bollingerUpperIndicatorName, bollingerStdIndicatorName, macdIndicatorName,
                    macdHistogramIndicatorName, macdSignalIndicatorName, newDailylyEMAShortIndicatorName,
                    newDailyEMALongIndicatorName, newRsiIndicatorName, newMomIndicatorName, newRocIndicatorName,newSmaIndicatorName,
                    newBollingerLowerIndicatorName, newBollingerMiddleIndicatorName, newBollingerUpperIndicatorName,
                    newBollingerStdIndicatorName, newMACDIndicatorName, newMACDHistogramIndicatorName,
                    newMACDSignalIndicatorName)


    # Convert features into rolling z-scores
    algoData = Zscore(algoData, featuresRaw, rollingWindow)

    # Normalize features by taking features delta divided by closing price
    algoData = Normalize(algoData, featuresRaw, deltaDays)

    # Drop NaN's
    algoData = algoData.dropna()
    algoDataDf = convert_df(algoData)
    
    
    df = algoData.copy()
    
    
    
    
    # Buy and Hold Strategy Returns
    @st.cache(allow_output_mutation=True)
    def buyHoldReturns(algoData):

        # Make first return and signal 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0
        algoData["Signal"][0] = 0

        algoData['cumBuyHoldReturns'] = (1+algoData['ActualReturns']).cumprod()
        
        returns = algoData[['Signal', "ActualReturns", 'cumBuyHoldReturns']]
        
        return returns
    
    
    
    # MACD Strategy Function
    @st.cache(allow_output_mutation=True)
    def macdStrategy(algoData, risk):
        
        signal = [0]
        position = False
        price = 0
        
        for ind in range(1, algoData.shape[0]):
        
            if((algoData['MACDline'][ind] > algoData['MACDSignal'][ind]) & (algoData['MACDline'][ind-1] > algoData['MACDSignal'][ind-1])):
            
                signal.append(1)
                position = True
                price = algoData['Close'][ind]
        
            elif ((algoData['MACDline'][ind] < algoData['MACDSignal'][ind]) & (algoData['MACDline'][ind-1] < algoData['MACDSignal'][ind-1])):

                signal.append(-1)
                position = True
                price = algoData['Close'][ind]
                
            elif ((position == True) & (signal[-1] == 1) & (algoData['Close'][ind] < price*(1-risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]
            elif((position == True) & (signal[-1] == -1) & (algoData['Close'][ind] > price*(1+risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]
                
            else:
            
                signal.append(np.nan)
            
        return signal
    
    
    # MACD Strategy returns
    @st.cache(allow_output_mutation=True)
    def macdReturns(algoData, risk):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['MACDStrategy'] = macdStrategy(algoData, risk)
        algoData['MACDStrategy'] = algoData['MACDStrategy'].ffill()
        algoData['MACDStrategyReturns'] = algoData['ActualReturns'] * algoData['MACDStrategy']
        algoData['MACDStrategyReturns'] = algoData['MACDStrategyReturns'].fillna(0)
        algoData['cumMACDReturns'] = (1 + algoData['MACDStrategyReturns']).cumprod()
        
        returns = algoData[['MACDStrategy', 'MACDStrategyReturns', 'cumMACDReturns']]
        
        return returns
    
    
    
    # RSI Strategy Function
    @st.cache(allow_output_mutation=True)
    def rsiSystem(algoData, risk, overbought, oversold, neutral):
        
        signal = [0]
        position = False
        price = 0
        

        for ind in range(1, algoData.shape[0]):

            if ((algoData['RSIline'][ind] > overbought) & (algoData['RSIline'][ind-1] < overbought) & (position == False)):
                signal.append(-1)
                position = True
                price = algoData['Close'][ind]
            elif ((algoData['RSIline'][ind] < oversold) & (algoData['RSIline'][ind-1] > oversold) & (position == False)):
                signal.append(1)
                position = True
                price = algoData['Close'][ind]
            elif ((algoData['RSIline'][ind] < neutral) & (algoData['RSIline'][ind-1] > neutral) & (position == True) & (signal[-1]==1)):
                signal.append(0)
                position = False
                price = algoData['Close'][ind]
            elif ((algoData['RSIline'][ind] > neutral) & (algoData['RSIline'][ind-1] < neutral) & (position == True) & (signal[-1]==-1)):
                signal.append(0)
                position = False
                price = algoData['Close'][ind]
            elif ((position == True) & (signal[-1] == 1) & (algoData['Close'][ind] < price*(1-risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]
            
            elif ((position == True) & (signal[-1] == -1) & (algoData['Close'][ind] > price*(1+risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]   
                
            else:
                signal.append(np.nan)
                
                
        return signal
    
    
    
    # RSI Strategy returns
    @st.cache(allow_output_mutation=True)
    def rsiReturns(algoData, risk):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['RSIStrategy'] = rsiSystem(algoData, risk, overbought, oversold, neutral)
        algoData['RSIStrategy'] = algoData['RSIStrategy'].ffill()
        algoData['RSIStrategyReturns'] = algoData['ActualReturns'] * algoData['RSIStrategy']
        algoData['RSIRayStrategyReturns'] = algoData['RSIStrategyReturns'].fillna(0)
        algoData['cumRSIReturns'] = (1 + algoData['RSIStrategyReturns']).cumprod()
        
        returns = algoData[['RSIStrategy', 'RSIStrategyReturns', 'cumRSIReturns']]
        
        return returns
    
    
    
    
    
    # Impulse System Function
    @st.cache(allow_output_mutation=True)
    def impulseSystem(algoData, risk):
        
        signal = [0]
        position = False
        price = 0
        
        for ind in range(1, algoData.shape[0]):
        
            if((algoData['EMAShort'][ind] > algoData['EMAShort'][ind-1]) & (algoData['EMALong'][ind] > algoData['EMALong'][ind-1]) & (algoData['MACDHistogram'][ind] > algoData['MACDHistogram'][ind-1])):
            
                signal.append(1)
                position = True
                price=algoData['Close'][ind]
        
            elif ((algoData['EMAShort'][ind] < algoData['EMAShort'][ind-1]) & (algoData['EMALong'][ind] < algoData['EMALong'][ind-1]) & (algoData['MACDHistogram'][ind] < algoData['MACDHistogram'][ind-1])):

                signal.append(-1)
                position = True
                price=algoData['Close'][ind]
                
                
                
            elif ((position == True) & (signal[-1] == 1) & (algoData['Close'][ind] < price*(1-risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]
            
            elif ((position == True) & (signal[-1] == -1) & (algoData['Close'][ind] > price*(1+risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]   
        
            else:
            
                signal.append(np.nan)
            
        return signal
    
    
    # Impulse Strategy returns
    @st.cache(allow_output_mutation=True)
    def impulseReturns(algoData, risk):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['ImpulseStrategy'] = impulseSystem(algoData, risk)
        algoData['ImpulseStrategy'] = algoData['ImpulseStrategy'].ffill()
        algoData['ImpulseStrategyReturns'] = algoData['ActualReturns'] * algoData['ImpulseStrategy']
        algoData['ImpulseStrategyReturns'] = algoData['ImpulseStrategyReturns'].fillna(0)
        algoData['cumImpulseReturns'] = (1 + algoData['ImpulseStrategyReturns']).cumprod()
        
        returns = algoData[['ImpulseStrategy', 'ImpulseStrategyReturns', 'cumImpulseReturns']]
        
        return returns
    
    
    
    
    
    # Bollinger Bands Function
    @st.cache(allow_output_mutation=True)
    def bbStrategy(algoData, risk):
        
        signal = [0]
        position = False
        price=0
        
        for ind in range(1, algoData.shape[0]):
            
            if ((algoData['Close'][ind] > algoData['upperBB'][ind]) & (algoData['Close'][ind-1] < algoData['upperBB'][ind-1]) & (position == False)):
                signal.append(-1)
                position = True
                price = algoData['Close'][ind]
            elif ((algoData['Close'][ind] < algoData['lowerBB'][ind]) & (algoData['Close'][ind-1] > algoData['lowerBB'][ind-1]) & (position == False)):
                signal.append(1)
                position = True
                price = algoData['Close'][ind]
            elif ((algoData['Close'][ind] < algoData['middleBB'][ind]) & (algoData['Close'][ind-1] > algoData['middleBB'][ind-1]) & (position == True)):
                signal.append(0)
                position = False
                price = algoData['Close'][ind]
            elif ((algoData['Close'][ind] > algoData['middleBB'][ind]) & (algoData['Close'][ind-1] < algoData['middleBB'][ind-1]) & (position == True)):
                signal.append(0)
                position = False
                price = algoData['Close'][ind]
                
            elif ((position == True) & (signal[-1] == 1) & (algoData['Close'][ind] < price*(1-risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]
            
            elif ((position == True) & (signal[-1] == -1) & (algoData['Close'][ind] > price*(1+risk))):
                position = False
                signal.append(0)
                price = algoData['Close'][ind]  
            else:
                signal.append(np.nan)
        
        return signal
    
    
    
    # Bollinger Bands Strategy returns
    @st.cache(allow_output_mutation=True)
    def bollingerReturns(algoData, risk):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        # Caluclate BB Strategy
        algoData['BBStrategy'] = bbStrategy(algoData, risk)
        algoData['BBStrategy'] = algoData['BBStrategy'].ffill()
        algoData['BBStrategyReturns'] = algoData['ActualReturns'] * algoData['BBStrategy']
        algoData['BBStrategyReturns'] = algoData['BBStrategyReturns'].fillna(0)
        algoData['cumBBReturns'] = (1 + algoData['BBStrategyReturns']).cumprod()

        returns = algoData[['BBStrategy', 'BBStrategyReturns', 'cumBBReturns']]
        
        return returns
    
    
    # Calculate all Strategy Returns and place in a dataframe 
    @st.cache(allow_output_mutation=True)
    def allReturnsData(algoData, risk):
        
        buyHoldReturn = buyHoldReturns(algoData)
        macdReturn = macdReturns(algoData, risk)
        rsiReturn = rsiReturns(algoData, risk)
        impulseReturn = impulseReturns(algoData, risk)
        bollingerReturn = bollingerReturns(algoData, risk)
        
        allReturns = pd.concat([buyHoldReturn, macdReturn, rsiReturn, impulseReturn, bollingerReturn], axis=1)
        allReturns = allReturns.dropna()
        
        return allReturns
    
    
    
    
    # Plot Strategy Returns
    @st.cache(allow_output_mutation=True)
    def cumulativeStrategyReturnsPlot(allStrategyReturns, assetName):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumBuyHoldReturns'],
                name="Buy&Hold",
                line=dict(color="green")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumMACDReturns'],
                name='MACD',
                line=dict(color="red")
            ))


        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumRSIReturns'],
                name='RSI',
                line=dict(color="blue")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumImpulseReturns'],
                name='Impulse System',
                line=dict(color="orange")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumBBReturns'],
                name='Bollinger Bands',
                line=dict(color="purple")
            ))



        fig.update_layout(
            title={
                'text': "Strategy Performance",
            },
            width=650, 
            height=430,
            template='seaborn',
            paper_bgcolor='#F8F8FF',
            plot_bgcolor='white',
            xaxis=dict(autorange=True,
                    title_text='Date',
                    showline=True,
                    linecolor='grey',
                    linewidth=1,
                    mirror=True,
                    ticks='outside',
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgrey'),
            yaxis=dict(autorange=True,
                    title_text='Cumulative Returns',
                    showline=True,
                    linecolor='grey',
                    linewidth=1,
                    mirror=True,
                    ticks='outside',
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgrey',
                    ),
            legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.025,
                    xanchor='right',
                    x=1
        ))
        
        return fig
    
    
    
    
    # Descriptive Statistics Function
    @st.cache(allow_output_mutation=True)
    def descriptiveStatsTable(allStrategyReturns, initial, riskFree, assetName):

        # Calculate Descriptive Statistics

        start_date = allStrategyReturns.index.min()
        end_date = allStrategyReturns.index.max()

        start = str(start_date.day)+'-'+str(start_date.month)+'-'+str(start_date.year)
        end = str(end_date.day)+'-'+str(end_date.month)+'-'+str(end_date.year)

        days = (end_date - start_date).days
        years = days/365

        init_investment = initial
        rf = riskFree

        buyHold_start = init_investment
        macd_start = init_investment
        rsi_start = init_investment
        impulse_start = init_investment
        bollinger_start = init_investment

        buyHold_end = round(allStrategyReturns['cumBuyHoldReturns'][-1] * init_investment,2)
        macd_end = round(allStrategyReturns['cumMACDReturns'][-1] * init_investment,2)
        rsi_end = round(allStrategyReturns['cumRSIReturns'][-1] * init_investment,2)
        impulse_end = round(allStrategyReturns['cumImpulseReturns'][-1] * init_investment,2)
        bollinger_end = round(allStrategyReturns['cumBBReturns'][-1] * init_investment,2)

        buyHold_max_dailyReturn = round(allStrategyReturns['ActualReturns'].max(),6)
        macd_max_dailyReturn = round(allStrategyReturns['MACDStrategyReturns'].max(),6)
        rsi_max_dailyReturn = round(allStrategyReturns['RSIStrategyReturns'].max(),6)
        impulse_max_dailyReturn = round(allStrategyReturns['ImpulseStrategyReturns'].max(),6)
        bollinger_max_dailyReturn = round(allStrategyReturns['BBStrategyReturns'].max(),6)

        buyHold_min_dailyReturn = round(allStrategyReturns['ActualReturns'].min(),6)
        macd_min_dailyReturn = round(allStrategyReturns['MACDStrategyReturns'].min(),6)
        rsi_min_dailyReturn = round(allStrategyReturns['RSIStrategyReturns'].min(),6)
        impulse_min_dailyReturn = round(allStrategyReturns['ImpulseStrategyReturns'].min(),6)
        bollinger_min_dailyReturn = round(allStrategyReturns['BBStrategyReturns'].min(),6)

        buyHold_max_drawdown = round(((allStrategyReturns['cumBuyHoldReturns'].min() - allStrategyReturns['cumBuyHoldReturns'].max())/allStrategyReturns['cumBuyHoldReturns'].max()),6)
        macd_max_drawdown = round(((allStrategyReturns['cumMACDReturns'].min() - allStrategyReturns['cumMACDReturns'].max())/allStrategyReturns['cumMACDReturns'].max()),6)
        rsi_max_drawdown = round(((allStrategyReturns['cumRSIReturns'].min() - allStrategyReturns['cumRSIReturns'].max())/allStrategyReturns['cumRSIReturns'].max()),6)
        impulse_max_drawdown = round(((allStrategyReturns['cumImpulseReturns'].min() - allStrategyReturns['cumImpulseReturns'].max())/allStrategyReturns['cumImpulseReturns'].max()),6)
        bollinger_max_drawdown = round(((allStrategyReturns['cumBBReturns'].min() - allStrategyReturns['cumBBReturns'].max())/allStrategyReturns['cumBBReturns'].max()),6)
        
        buyHoldSignals = allStrategyReturns.Signal[(allStrategyReturns['Signal'] == 1) | (allStrategyReturns['Signal'] == -1)].count()
        macdSignals = allStrategyReturns.MACDStrategy[(allStrategyReturns['MACDStrategy'] == 1) | (allStrategyReturns['MACDStrategy'] == -1)].count()
        rsiSignals = allStrategyReturns.RSIStrategy[(allStrategyReturns['RSIStrategy'] == 1) | (allStrategyReturns['RSIStrategy'] == -1)].count()
        impulseSignals = allStrategyReturns.ImpulseStrategy[(allStrategyReturns['ImpulseStrategy'] == 1) | (allStrategyReturns['ImpulseStrategy'] == -1)].count()
        bollingerSignals = allStrategyReturns.BBStrategy[(allStrategyReturns['BBStrategy'] == 1) | (allStrategyReturns['BBStrategy'] == -1)].count()
        
        buyHoldPos = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] > 0)].count()
        macdPos = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] > 0)].count()
        rsiPos = allStrategyReturns.RSIStrategyReturns[(allStrategyReturns['RSIStrategyReturns'] > 0)].count()
        impulsePos = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] > 0)].count()
        bollingerPos = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] > 0)].count()
        
        buyHoldPosPerc = round((buyHoldPos/buyHoldSignals),6)
        macdPosPerc = round((macdPos/macdSignals),6)
        rsiPosPerc = round((rsiPos/rsiSignals),6)
        impulsePosPerc = round((impulsePos/impulseSignals),6)
        bollingerPosPerc = round((bollingerPos/bollingerSignals),6)
        
        buyHoldPosSum = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] > 0)].sum()
        macdPosSum = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] > 0)].sum()
        rsiPosSum = allStrategyReturns.RSIStrategyReturns[(allStrategyReturns['RSIStrategyReturns'] > 0)].sum()
        impulsePosSum = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] > 0)].sum()
        bollingerPosSum = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] > 0)].sum()
        
        buyHoldPosAvg = round((buyHoldPosSum/buyHoldPos),6)
        macdPosAvg = round((macdPosSum/macdPos),6)
        rsiPosAvg = round((rsiPosSum/rsiPos),6)
        impulsePosAvg = round((impulsePosSum/impulsePos),6)
        bollingerPosAvg = round((bollingerPosSum/bollingerPos),6)
        
        buyHoldNeg = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] < 0)].count()
        macdNeg = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] < 0)].count()
        rsiNeg = allStrategyReturns.RSIStrategyReturns[(allStrategyReturns['RSIStrategyReturns'] < 0)].count()
        impulseNeg = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] < 0)].count()
        bollingerNeg = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] < 0)].count()
        
        
        buyHoldNegPerc = round((buyHoldNeg/buyHoldSignals),6)
        macdNegPerc = round((macdNeg/macdSignals),6)
        rsiNegPerc = round((rsiNeg/rsiSignals),6)
        impulseNegPerc = round((impulseNeg/impulseSignals),6)
        bollingerNegPerc = round((bollingerNeg/bollingerSignals),6)
        
        
        buyHoldNegSum = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] < 0)].sum()
        macdNegSum = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] < 0)].sum()
        rsiNegSum = allStrategyReturns.RSIStrategyReturns[(allStrategyReturns['RSIStrategyReturns'] < 0)].sum()
        impulseNegSum = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] < 0)].sum()
        bollingerNegSum = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] < 0)].sum()
        
        buyHoldNegAvg = round((buyHoldNegSum/buyHoldNeg),6)
        macdNegAvg = round((macdNegSum/macdNeg),6)
        rsiNegAvg = round((rsiNegSum/rsiNeg),6)
        impulseNegAvg = round((impulseNegSum/impulseNeg),6)
        bollingerNegAvg = round((bollingerNegSum/bollingerNeg),6)
        
        buyHold_annualReturn = allStrategyReturns['ActualReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        macd_annualReturn = allStrategyReturns['MACDStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        rsi_annualReturn = allStrategyReturns['RSIStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        impulse_annualReturn = allStrategyReturns['ImpulseStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        bollinger_annualReturn = allStrategyReturns['BBStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1


        buyHold_annualVol = allStrategyReturns['ActualReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        macd_annualVol = allStrategyReturns['MACDStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        rsi_annualVol = allStrategyReturns['RSIStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        impulse_annualVol = allStrategyReturns['ImpulseStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        bollinger_annualVol = allStrategyReturns['BBStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)

        buyHold_Sharpe = round((buyHold_annualReturn-rf)/buyHold_annualVol,2)
        macd_Sharpe = round((macd_annualReturn-rf)/macd_annualVol,2)
        rsi_Sharpe = round((rsi_annualReturn-rf)/rsi_annualVol,2)
        impulse_Sharpe = round((impulse_annualReturn-rf)/impulse_annualVol,2)
        bollinger_Sharpe = round((bollinger_annualReturn-rf)/bollinger_annualVol,2)


        buyHold_variance = allStrategyReturns['ActualReturns'].var()

        macd_covariance = allStrategyReturns['MACDStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        rsi_covariance = allStrategyReturns['RSIStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        impulse_covariance = allStrategyReturns['ImpulseStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        bollinger_covariance = allStrategyReturns['BBStrategyReturns'].cov(allStrategyReturns['ActualReturns'])

        buyHold_beta = 1

        macd_beta = round(macd_covariance/buyHold_variance,2)
        rsi_beta = round(rsi_covariance/buyHold_variance,2)
        impulse_beta = round(impulse_covariance/buyHold_variance,2)
        bollinger_beta = round(bollinger_covariance/buyHold_variance,2)
        

        # Table of Descriptive Statistics
        
        head = ['<b>Statistic<b>', '<b>Buy&Hold<b>', '<b>MACD<b>', '<b>RSI<b>', '<b>Impulse<b>', '<b>Bollinger<b>']
        labels = ['Start Date', 'End Date','Initial Investment', 'Final Investment','--------------------------',
                'Signals', 'Winning Trades', 'Losing Trades', '% Winning', '% Losing', 
                'Average Profit','Average Loss', '--------------------------', 
                'Max Daily Return','Min Daily Return', 'Max Drawdown', 'Annual Return',
                'Annual Volatility', 'Sharpe Ratio', 'Beta']


        buyHold_stats = [start, end,'${:,}'.format(buyHold_start), '${:,}'.format(buyHold_end), '--------------------------',
                        buyHoldSignals, buyHoldPos, buyHoldNeg,'{:.2%}'.format(buyHoldPosPerc),'{:.2%}'.format(buyHoldNegPerc), 
                        '{:.2%}'.format(buyHoldPosAvg), '{:.2%}'.format(buyHoldNegAvg), '--------------------------', 
                        '{:.2%}'.format(buyHold_max_dailyReturn), '{:.2%}'.format(buyHold_min_dailyReturn), 
                        '{:.2%}'.format(buyHold_max_drawdown), '{:.2%}'.format(buyHold_annualReturn), 
                        '{:.2%}'.format(buyHold_annualVol), buyHold_Sharpe, buyHold_beta]


        macd_stats = [start, end,'${:,}'.format(macd_start), '${:,}'.format(macd_end),'--------------------------', 
                    macdSignals, macdPos, macdNeg,'{:.2%}'.format(macdPosPerc),'{:.2%}'.format(macdNegPerc),
                    '{:.2%}'.format(macdPosAvg), '{:.2%}'.format(macdNegAvg), '--------------------------', 
                    '{:.2%}'.format(macd_max_dailyReturn), '{:.2%}'.format(macd_min_dailyReturn), 
                    '{:.2%}'.format(macd_max_drawdown), '{:.2%}'.format(macd_annualReturn), 
                    '{:.2%}'.format(macd_annualVol), macd_Sharpe, macd_beta]

        rsi_stats = [start, end, '${:,}'.format(rsi_start), '${:,}'.format(rsi_end),'--------------------------', 
                        rsiSignals, rsiPos, rsiNeg, '{:.2%}'.format(rsiPosPerc),
                        '{:.2%}'.format(rsiNegPerc), '{:.2%}'.format(rsiPosAvg), '{:.2%}'.format(rsiNegAvg), 
                        '--------------------------', '{:.2%}'.format(rsi_max_dailyReturn), 
                        '{:.2%}'.format(rsi_min_dailyReturn), '{:.2%}'.format(rsi_max_drawdown),
                        '{:.2%}'.format(rsi_annualReturn), '{:.2%}'.format(rsi_annualVol), 
                        rsi_Sharpe, rsi_beta]


        impulse_stats = [start, end, '${:,}'.format(impulse_start), '${:,}'.format(impulse_end), '--------------------------', 
                        impulseSignals, impulsePos, impulseNeg, '{:.2%}'.format(impulsePosPerc),'{:.2%}'.format(impulseNegPerc), 
                        '{:.2%}'.format(impulsePosAvg), '{:.2%}'.format(impulseNegAvg), '--------------------------', 
                        '{:.2%}'.format(impulse_max_dailyReturn), '{:.2%}'.format(impulse_min_dailyReturn), 
                        '{:.2%}'.format(impulse_max_drawdown), '{:.2%}'.format(impulse_annualReturn), 
                        '{:.2%}'.format(impulse_annualVol), impulse_Sharpe, impulse_beta]


        bollinger_stats = [start, end, '${:,}'.format(bollinger_start), '${:,}'.format(bollinger_end), '--------------------------',
                        bollingerSignals, bollingerPos, bollingerNeg,'{:.2%}'.format(bollingerPosPerc),
                        '{:.2%}'.format(bollingerNegPerc), '{:.2%}'.format(bollingerPosAvg), 
                        '{:.2%}'.format(bollingerNegAvg), '--------------------------', '{:.2%}'.format(bollinger_max_dailyReturn), 
                        '{:.2%}'.format(bollinger_min_dailyReturn), '{:.2%}'.format(bollinger_max_drawdown), 
                        '{:.2%}'.format(bollinger_annualReturn), '{:.2%}'.format(bollinger_annualVol), 
                        bollinger_Sharpe, bollinger_beta]




        fig11 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
            cells=dict(values=[labels, buyHold_stats, macd_stats, rsi_stats, impulse_stats, bollinger_stats],
                fill_color='lavender',
                align='left'))
        ])

        fig11.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=650, height=430)
        
        return fig11

   
    # Split the data into training and test datasets

    # Assign a copy of the aldoData DataFrame
    returns = algoData[algoData.columns]

    # Select the start of the training period
    returns_begin = returns.index.min()

    # Select the ending period for the training data with an offset of x months
    returns_end = returns.index.min() + DateOffset(months=training_months)

    # Generate the taining DataFrames
    returns_train = returns.loc[returns_begin:returns_end]

    # Generate the test DataFrames
    returns_test = returns.loc[returns_end+DateOffset(hours=1):]
    
    # Calulate all the Strategy returns for the test period
    allStrategyReturnsTest = allReturnsData(returns_test, risk)
    allStrategyReturnsTestDf = convert_df(allStrategyReturnsTest)
    
    #results = descriptiveStatsTable(allStrategyReturnsTest, initial, riskFree, assetNames)
        
    # Plot the all the Strategy returns for the test period
    cumulativeStrategyReturnsPlotTestPeriod = cumulativeStrategyReturnsPlot(allStrategyReturnsTest, assetNames)

    # Descriptive Statistics for all Strategy returns for the test period
    descriptiveStatsTestPeriod = descriptiveStatsTable(allStrategyReturnsTest, initial, riskFree, assetNames)
    
    
    ################################################################################################################################
    
    # raw Features
    rawFeatures = ['EMAShort', 'EMALong',  'RSIline', 'MOMline', 'ROCline','SMAline', 'lowerBB_zscore', 'middleBB',
               "upperBB", "2stdBB", "MACDline", 'MACDHistogram','MACDSignal']


    # z-scores Features
    zScoreFeatures = ['EMAShort_zscore', 'EMALong_zscore',  'RSIline_zscore', 'MOMline_zscore', 'ROCline_zscore',
                   'SMAline_zscore', 'lowerBB_zscore', 'middleBB_zscore', "upperBB_zscore", 
                  "2stdBB_zscore", "MACDline_zscore", 'MACDHistogram_zscore','MACDSignal_zscore' ]


    # normal Features
    normalFeatures = ['EMAShort_normal', 'EMALong_normal',  'RSIline_normal', 'MOMline_normal', 'ROCline_normal',
                   'SMAline_normal', 'lowerBB_normal', 'middleBB_normal', "upperBB_normal", 
                  "2stdBB_normal", "MACDline_normal", 'MACDHistogram_normal','MACDSignal_normal' ]

    featureList = [rawFeatures, zScoreFeatures, normalFeatures]
    
    
    # Calculate Descriptive Statistics for Random Forest Model
    @st.cache(allow_output_mutation=True) 
    def descriptiveRFStatistics(predictions_df, model):

        start_date = predictions_df.index.min()
        end_date = predictions_df.index.max()

        start = str(start_date.day)+'-'+str(start_date.month)+'-'+str(start_date.year)
        end = str(end_date.day)+'-'+str(end_date.month)+'-'+str(end_date.year)

        days = (end_date - start_date).days
        years = days/365

        init_investment = initial
        rf = riskFree

        RFstrategy_start = init_investment

        RFstrategy_end = round(predictions_df['cumStrategyReturns'][-1] * init_investment,2)

        RFstrategy_max_dailyReturn = round(predictions_df['StrategyReturns'].max(),6)

        RFstrategy_min_dailyReturn = round(predictions_df['StrategyReturns'].min(),6)

        RFstrategy_max_drawdown = round(((predictions_df['cumStrategyReturns'].min() - predictions_df['cumStrategyReturns'].max())/predictions_df['cumStrategyReturns'].max()),6)

        RFstrategyPredicted = predictions_df.Predicted[(predictions_df['Predicted'] == 1) | (predictions_df['Predicted'] == -1)].count()

        RFstrategyPos = predictions_df.StrategyReturns[(predictions_df['StrategyReturns'] > 0)].count()

        RFstrategyPosPerc = round((RFstrategyPos/RFstrategyPredicted),6)

        RFstrategyPosSum = predictions_df.StrategyReturns[(predictions_df['StrategyReturns'] > 0)].sum()

        RFstrategyPosAvg = round((RFstrategyPosSum/RFstrategyPos),6)

        RFstrategyNeg = predictions_df.StrategyReturns[(predictions_df['StrategyReturns'] < 0)].count()

        RFstrategyNegPerc = round((RFstrategyNeg/RFstrategyPredicted),6)

        RFstrategyNegSum = predictions_df.StrategyReturns[(predictions_df['StrategyReturns'] < 0)].sum()

        RFstrategyNegAvg = round((RFstrategyNegSum/RFstrategyNeg),6)

        RFstrategy_annualReturn = predictions_df['StrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1

        RFstrategy_annualVol = predictions_df['StrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)

        RFstrategy_Sharpe = round((RFstrategy_annualReturn-rf)/RFstrategy_annualVol,2)

        market_variance = predictions_df['ActualReturns'].var()

        RFstrategy_covariance = predictions_df['StrategyReturns'].cov(predictions_df['ActualReturns'])

        RFstrategy_beta = round(RFstrategy_covariance/market_variance,2)

        # Table of Descriptive Statistics

        labels = ['Start Date', 'End Date','Initial Investment', 'Final Investment','--------------------------',
                'Signals', 'Winning Trades', 'Losing Trades', '% Winning', '% Losing', 
                'Average Profit','Average Loss', '--------------------------', 
                'Max Daily Return','Min Daily Return', 'Max Drawdown', 'Annual Return',
                'Annual Volatility', 'Sharpe Ratio', 'Beta']


        RFstrategy_stats = [start, end,'${:,}'.format(RFstrategy_start), '${:,}'.format(RFstrategy_end), '--------------------------',
                        RFstrategyPredicted, RFstrategyPos, RFstrategyNeg,'{:.2%}'.format(RFstrategyPosPerc),
                        '{:.2%}'.format(RFstrategyNegPerc), '{:.2%}'.format(RFstrategyPosAvg), '{:.2%}'.format(RFstrategyNegAvg), 
                            '--------------------------', '{:.2%}'.format(RFstrategy_max_dailyReturn),
                        '{:.2%}'.format(RFstrategy_min_dailyReturn), '{:.2%}'.format(RFstrategy_max_drawdown), 
                        '{:.2%}'.format(RFstrategy_annualReturn), '{:.2%}'.format(RFstrategy_annualVol),
                            RFstrategy_Sharpe, RFstrategy_beta]


        metrics = {'Statistic': labels, model: RFstrategy_stats}

        results = pd.DataFrame(data=metrics)
        results = results.set_index('Statistic')
        
        return results
    
    
    def descriptiveRFStats(results, assetName):
    
        # Table of Descriptive Statistics

        head = ['<b>Statistic<b>', '<b>standardScaler<b>', '<b>zScores<b>', '<b>normalized<b>']

        fig11 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
            cells=dict(values=[results.index, results['standardScaler'], results['zScores'], results['normalized']],
                fill_color='lavender',
                align='left'))
        ])

        fig11.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=650, height=430)
        
        return fig11
    
    

    # Plot Strategy Returns
    @st.cache(allow_output_mutation=True)
    def cumulativeRFStrategyReturnsPlot(allStrategyReturns, assetName):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['AssetReturns'],
                name="Buy&Hold",
                line=dict(color="green")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['standardScaler'],
                name='standardScaler',
                line=dict(color="red")
            ))


        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['zScores'],
                name='zScores',
                line=dict(color="blue")
            ))


        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['normalized'],
                name='normalized',
                line=dict(color="purple")
            ))



        fig.update_layout(
            title={
                'text': "Model Performance",
            },
            width=650,
            height=430,
            template='seaborn',
            paper_bgcolor='#F8F8FF',
            plot_bgcolor='white',
            xaxis=dict(autorange=True,
                    title_text='Date',
                    showline=True,
                    linecolor='grey',
                    linewidth=1,
                    mirror=True,
                    ticks='outside',
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgrey'),
            

            yaxis=dict(autorange=True,
                    title_text='Cumulative Returns',
                    showline=True,
                    linecolor='grey',
                    linewidth=1,
                    mirror=True,
                    ticks='outside',
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgrey',
                    ),
            legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.025,
                    xanchor='right',
                    x=0.9
        ))

        
        return fig
    
    
    rfModelScores = []
    rfModelPerformance = []
    cumStrategyReturns = []
    
    for var in range(0,3):
        
        # Separate the y variable and the features
        y = df['Signal'].copy()
        X = df[featureList[var]]
        
        # Split data into training and test sets
        
        training_begin = X.index.min()
        training_end = X.index.min() + DateOffset(months=training_months)
        
        # Generate the X_train and y_train DataFrames
        X_train = X.loc[training_begin:training_end]
        y_train = y.loc[training_begin:training_end]
        
        # Generate the X_test and y_test DataFrames
        X_test = X.loc[training_end+DateOffset(days=1):]
        y_test = y.loc[training_end+DateOffset(days=1):]

        ind = X_test.index
        
        if(var == 0):
    
            # Create a StandardScaler instance
            scaler = StandardScaler()
 
            # Apply the scaler model to fit the X-train data
            X_scaler = scaler.fit(X_train)

            X_train = X_scaler.transform(X_train)
            X_test = X_scaler.transform(X_test)
            
        # Create the random forest classifier instance
        rf_model = RandomForestClassifier(n_estimators=estimators,  max_depth=depth, min_samples_split=samples_split,
                                          min_samples_leaf=samples_leaf, random_state=42)
    
        # Fit the model
        rf_model = rf_model.fit(X_train, y_train)
        
        # Making predictions using the testing data
        predictions = rf_model.predict(X_test)
        
        
        if(var == 0):
            model = 'standardScaler'
        elif(var==1):
            model = 'zScores'
        else:
            model = 'normalized'
        
        
        report = classification_report(y_test, predictions, output_dict=True)
        #rfMetrics = pd.DataFrame(report).transpose()
        rfMetrics = pd.DataFrame(report).transpose()
        rfMetrics['model'] = model 
        rfMetrics['scores'] = rfMetrics.index
        rfMetrics = rfMetrics.reset_index()
        
        rfModelScores.append(rfMetrics)
        
        # Create a predictions DataFrame
        predictions_df = pd.DataFrame(index=ind)

        # Add the RF model predictions to the DataFrame
        predictions_df['Predicted'] = predictions

        predictions_df['ActualReturns'] = df['ActualReturns']
        predictions_df['cumActualReturns'] = (1 + predictions_df['ActualReturns']).cumprod()

        # Add the strategy returns to the DataFrame
        predictions_df['StrategyReturns'] = predictions_df['ActualReturns'] * predictions_df['Predicted']
        # Add cumulative strategy returns to the DataFrame
        predictions_df['cumStrategyReturns'] = (1 + predictions_df['StrategyReturns']).cumprod()
        
        cumStrategyReturns.append(predictions_df)
        
        stats = descriptiveRFStatistics(predictions_df,model)
        
        rfModelPerformance.append(stats)
        
    # display rfScores
    rfScores = pd.concat(rfModelScores, axis=0)
    rfScores = rfScores.set_index(['model','scores'])
    rfScores = rfScores.drop(['index'], axis=1)
    
    rfScoresDf = convert_df(rfScores)

    # display cumulative strategy returns
    StrategyReturns = pd.concat(cumStrategyReturns, axis=1)
    StrategyReturns['AssetReturns'] = StrategyReturns.iloc[:,2]
    StrategyReturns['standardScaler'] = StrategyReturns.iloc[:,4]
    StrategyReturns['zScores'] = StrategyReturns.iloc[:,9]
    StrategyReturns['normalized'] = StrategyReturns.iloc[:,14]
    StrategyReturns = StrategyReturns.drop(['ActualReturns', 'cumActualReturns', 'cumStrategyReturns',
                                        'StrategyReturns'], axis=1)
    
    
    StrategyReturns['pred_standard'] = StrategyReturns.iloc[:,0]
    StrategyReturns['pred_zscores'] = StrategyReturns.iloc[:,1]
    StrategyReturns['pred_normalized'] = StrategyReturns.iloc[:,2]
    
    StrategyReturns = StrategyReturns.drop(['Predicted'], axis=1)
    
    StrategyReturnsDf = convert_df(StrategyReturns)

    st.markdown(' ')
    rfCumStrategyReturns = cumulativeRFStrategyReturnsPlot(StrategyReturns, assetNames)
    
    
    # display rfModelPerformance
    rfModelPerf = pd.concat(rfModelPerformance, axis=1)  
    
    rfdescriptiveStats = descriptiveRFStats(rfModelPerf, assetNames)
    
    
    #################################################################
    
    pctAssetReturns = pctReturns(algoData, assetNames)
    algoSignals = algoSignals(allStrategyReturnsTest, assetNames)
    rfSignals = rfSignals(StrategyReturns, assetNames)
    
    ret, algo, rf = st.columns([1,1,1])
    
    
    ret.plotly_chart(pctAssetReturns, use_column_width=True)
    
    
    algo.plotly_chart(algoSignals, use_column_width=True)
    
    rf.plotly_chart(rfSignals,use_column_width=True)
    
    
    #d.plotly_chart(rfdescriptiveStats, use_column_width=True)
    ##############################################################
    
    
    
    
    

    st.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Technical Indicators - Trading Strategies - Data Download<b></h3>", unsafe_allow_html=True) 
    
    
    with st.expander('Technical Indicators - Trading Strategies Descriptions'):
        st.markdown(' ')
        st.markdown("<h5 style='text-align: left; color: #872657; padding-left: 0px; font-size: 25px'><b>Technical Indicators<b></h5>", unsafe_allow_html=True)
        
    
        st.markdown("<h6 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Exponential Moving Average (EMA)<b></h6>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>An exponential moving average (EMA) is a type of moving average (MA) that places a greater 
                    weight and significance on the most recent data points. The exponential moving average is also referred to as the
                    exponentially weighted moving average. An exponentially weighted moving average reacts more significantly to recent 
                    price changes than a simple moving average (SMA), which applies an equal weight to all observations 
                    in the period.</p>''', unsafe_allow_html=True)
        
        
        st.markdown(' ')
        st.markdown("<h6 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Rate Of Change Index (ROC)<b></h6>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>The rate of change (ROC) is the speed at which a variable changes over a specific period of time. ROC is often 
                    used when speaking about momentum, and it can generally be expressed as a ratio between a change in one variable relative to a corresponding 
                    change in another; graphically, the rate of change is represented by the slope of a line.</p>''', unsafe_allow_html=True)
        
        
        st.markdown(' ')
        st.markdown("<h5 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Momentum Indicator (MOM)<b></h5>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>The Momentum indicator is a speed of movement indicator that is designed to identify the speed (or strength) of 
                    price movement. This indicator compares the current close price to the close price N bars ago and also displays a moving average of this 
                    difference.</p>''', unsafe_allow_html=True)
        
        
        
        
        st.markdown(' ')
        st.markdown("<h5 style='text-align: left; color: #872657; padding-left: 0px; font-size: 25px'><b>Trading Strategies<b></h5>", unsafe_allow_html=True)
        
    
        st.markdown("<h5 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>MACD<b></h5>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>Moving average convergence/divergence (MACD, or MAC-D) is a trend-following momentum indicator 
                    that shows the relationship between two exponential moving averages (EMAs) of a security’s price. The MACD line is calculated 
                    by subtracting the 26-period EMA from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of 
                    the MACD line is called the signal line, which is then plotted on top of the MACD line, which can function as a trigger for 
                    buy or sell signals. Traders may buy the security when the MACD line crosses above the signal line and sell—or short—the 
                    security when the MACD line crosses below the signal line.</p>''', unsafe_allow_html=True)
    
    
    
        st.markdown(' ')
        st.markdown("<h6 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Relative Strength Index (RSI)<b></h6>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and 
                    magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions in the price of that security. The RSI is 
                    displayed as an oscillator (a line graph) on a scale of zero to 100. Traditionally, an RSI reading of 70 or above indicates an overbought 
                    situation. A reading of 30 or below indicates an oversold condition.</p>''', unsafe_allow_html=True)
    

        
        st.markdown(' ')
        st.markdown("<h6 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Impulse System<b></h6>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>The Impulse System is based on two indicators, a 13-day EMA and the MACD-Histogram. 
                    The moving average identifies the trend, while the MACD-Histogram measures momentum. As a result, the Impulse System combines trend following and momentum to 
                    identify impulses that can be traded.</p>''', unsafe_allow_html=True)
        
        
        st.markdown(' ')
        st.markdown("<h5 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Bollinger Bands<b></h5>", unsafe_allow_html=True)
        st.markdown('''<p style='font-size: 15px'>A Bollinger Band is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) 
                    away from a simple moving average (SMA) of a security's price.</p>''', unsafe_allow_html=True)
        
        st.markdown(' ')
        st.markdown("<h5 style='text-align: left; color: #872657; padding-left: 0px; font-size: 25px'><b>References<b></h5>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>The New Trading For A Living - Author: Dr. Alexander Elder<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Pandas-TA Library - https://github.com/twopirllc/pandas-ta<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Investopedia - https://www.investopedia.com/terms/e/ema.asp<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Investopedia - https://www.investopedia.com/terms/r/rateofchange.asp<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Investopedia - https://www.investopedia.com/terms/m/macd.asp<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Investopedia - https://www.investopedia.com/terms/r/rsi.asp<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Investopedia - https://www.investopedia.com/terms/b/bollingerbands.asp<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Tradingview - https://www.tradingview.com/script/vv3r5BKm-MTM-Momentum-Indicator<b></i>", unsafe_allow_html=True)
        st.markdown("<i style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 15px'><b>Tradingview - https://www.tradingview.com/script/oCbEFfpg-Indicator-Elder-Impulse-System<b></i>", unsafe_allow_html=True)
    
    
    colg, colh, coli, colj = st.columns([1, 1, 1, 1])
    
    colg.download_button(label="algoData", data=algoDataDf, file_name='algoData_'+assetNames+'_.csv', mime='text/csv')
    colh.download_button(label="Strategy Results", data=allStrategyReturnsTestDf, file_name='strategyResults_'+assetNames+'.csv', mime='text/csv')
    coli.download_button(label="Model Results", data=StrategyReturnsDf, file_name='modelResults_'+assetNames+'.csv', mime='text/csv')
    colj.download_button(label="Model Scores", data=rfScoresDf, file_name='RFmodelScores_'+assetNames+'.csv', mime='text/csv')
    
    
    #st.markdown(" ")
    #st.markdown(" ")
    
    #st.markdown("<h4 style='text-align: left; color: teal; padding-left: 0px; font-size: 50px'><b>"+assetNames+"<b></h4>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Cumultive Returns<b></h4>", unsafe_allow_html=True)
   
    a, b = st.columns([1,1])
    
    #st.markdown(' ')
    a.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Trading Strategies<b></h4>", unsafe_allow_html=True)
    #st.markdown(' ')
    a.plotly_chart(cumulativeStrategyReturnsPlotTestPeriod, use_column_width=True)
    b.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Random Forest<b></h4>", unsafe_allow_html=True)
    b.plotly_chart(rfCumStrategyReturns, use_column_width=True)
    
    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    st.markdown("<h4 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Performance Statistics<b></h4>", unsafe_allow_html=True)
    
    c, d = st.columns([1,1])
    c.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Trading Strategies<b></h4>", unsafe_allow_html=True)
    #st.plotly_chart(rfCumStrategyReturns)
    c.plotly_chart(descriptiveStatsTestPeriod, use_column_width=True)
    d.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Random Forest<b></h4>", unsafe_allow_html=True)
    d.plotly_chart(rfdescriptiveStats, use_column_width=True)
    
    #st.write(rfModelPerformance)
    
    #################################################################################################################################