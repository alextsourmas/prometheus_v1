import numpy as np
import pandas as pd
import math 
import xgboost as xgb
import matplotlib.pyplot as plt
import yfinance as yf
import ta 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")



def get_stock_data(stock: str, period: str, verbose=True):
    '''
    Function to get the initial stock data with basic columns
    Args: 
        stock(str): string with stock name
        period(str): string with period (1Y, 5Y, 10Y, 'max', for example..)
        verbose(bool): choose whether to add print statements
    Rtypes: 
        stock_dataframe (pd.DataFrame): stock dataframe with all historical data
    '''
    
    if verbose: print('\nGetting stock data for {}...'.format(stock))
    stock_dataframe = yf.Ticker(stock)
    stock_dataframe = stock_dataframe.history(period=period)
    # stock_dataframe = stock_dataframe.history(period='max')
    stock_dataframe = stock_dataframe.reset_index()
    stock_dataframe['Date'] = stock_dataframe['Date'].astype(str)
    stock_dataframe = stock_dataframe.reset_index(drop=True)
    # stock_dataframe['Ticker'] = stock
    if verbose: print('Loaded stock data into memory.')
    return stock_dataframe 


def simple_moving_average(stock_df: pd.DataFrame, close_col: str, window: int, fillna: True, verbose = True): 
    '''
    Generate a simple moving average on a dataframe - save to a new column 
    Args: 
        stock_df(pd.DataFrame): dataframe to use
        close_col(str): name of the close column 
        window(int): lookback window for the moving average
        fillna(bool): choose whether to fill na's (especially if the df isn't big enough for the lookback...itll
        use as many rows as are available instead)
        verbose(bool): choose whether to add prints to the function 
    '''
    if verbose: print('\nGenerating simple moving average for {} days...'.format(window))
    min_periods = 0 if fillna else window
    rolling_series = stock_df['Close'].rolling(window=window, min_periods=min_periods).mean()
    if verbose: print('SMA calcualted for time period {}.'.format(window))
    return rolling_series



def trend_analysis(stock_df: pd.DataFrame, window: int, sma_col: str, close_col: str, verbose=True): 
    '''
    Perform trend analysis 
    If closing price value leads its MA15 and MA15 is rising for last n days then trend is Uptrend i.e. trend signal is 1.
    If closing price value lags its MA15 and MA15 is falling for last n days then trend is Downtrend i.e. trend signal is 0.
    
    Args: 
        stock_df(pd.DataFrame) dataframe with stock info and moving average
        window(int): how long to use for a window for the trend analysis 
        sma_col(str): name of 15 day moving average col to use (can use a different one)
        close_col(str): name of the close column
        verbose(bool): add print statements
    Rtypes: 
        series(pd.Series): column with the new trend analysis in it 
    '''

    if verbose: print('\nGetting trend analysis...')
    
    stock_df['trend_analysis'] = ''
    
    for row in range(0, len(stock_df)): 
        #Set variables 
        current_close = stock_df[close_col].loc[row]
        current_ma = stock_df[sma_col].loc[row]
        slice_df = stock_df.loc[row - (window-1): row]
        monotonic_increasing = slice_df[sma_col].is_monotonic_increasing
        monotonic_decreasing = slice_df[sma_col].is_monotonic_decreasing
        #If conditions are met, set trend
        if (current_close > current_ma) and monotonic_increasing: 
            trend = 'up'
        elif (current_close < current_ma) and monotonic_decreasing: 
            trend = 'down' 
        else:
            trend = 'no'
        
        stock_df['trend_analysis'].loc[row] = trend

    series = stock_df['trend_analysis']
    stock_df = stock_df.drop(columns='trend_analysis', inplace=True)
    if verbose: print('Trend analysis complete.')
    return series 


def get_quantified_trend(stock_df: pd.DataFrame, close_col: str, trend_analysis_col: str, window=3, verbose=True):
    '''
    Quantify the trend variables based on the equation provided
    Args: 
        stock_df(pd.Dataframe) stock df to use
        close_col(str): name of close column
        trend_analysis_col(str): name of trend analysis column
        window(int): window size to use for the analysis (default is 3 according to the paper)
        verbose(bool): choose whether to add prints
    Rtypes: 
        stock_df(pd.DataFrame): finished dataframe with quantified trend column
    '''
    if verbose: print('\nCalculating stock quantified trend...')
    stock_df['quantified_trend'] = ''

    for row in range(0, len(stock_df)):

        slice_df = stock_df.loc[row - (window - 1): row]
        current_trend = stock_df[trend_analysis_col].loc[row]
        current_close = stock_df[close_col].loc[row]
        min_cp = slice_df[close_col].min()
        max_cp = slice_df[close_col].max()
        #Check if denominator is zero - if it is, return zero rather than throw an error
        if (max_cp - min_cp) != 0:
            value_if_uptrend_or_hold = ((current_close - min_cp)/(max_cp - min_cp) * 0.5) + 0.5
            value_if_downtrend = ((current_close - min_cp)/(max_cp - min_cp) * 0.5)
        else: 
            value_if_uptrend_or_hold = 0
            value_if_downtrend = 0
        #Set values based on condition
        if current_trend == 'up': 
            stock_df['quantified_trend'].loc[row] = value_if_uptrend_or_hold

        if current_trend == 'no': 
            stock_df['quantified_trend'].loc[row] = value_if_uptrend_or_hold 

        if current_trend == 'down': 
            stock_df['quantified_trend'].loc[row] = value_if_downtrend

    series = stock_df['quantified_trend']
    stock_df = stock_df.drop(columns='quantified_trend', inplace=True)
    if verbose: print('Calculated stock quantified trend.')
    return series


def train_xgb(stock_df: pd.DataFrame, train_set_percent: float, random_state: int, verbose=True): 
    '''
    Train XGB model and return test dataframe with predictions in it
    Args: 
        stock_df: Df with everything to train and test models on 
        train_set_percent: percentage of df to use for training
        random_state(int): set random seed for the model
    Rtypes: 
        final_test_set: test set df with predictions in it
    '''

    if verbose: print('\nSplitting out training and testing sets...')
    #Insert machine learning model here rather than just relying on rules - predict quantified trend 
    train_set, test_set= np.split(stock_df, [int(train_set_percent *len(stock_df))])

    train_dates_df = train_set[['Date']]
    x_train = train_set.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
    y_train = train_set[['quantified_trend']]

    test_dates_df = test_set[['Date']]
    x_test = test_set.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
    y_test = test_set[['quantified_trend']]

    #Fit a model 
    if verbose: print('Training XGBoost model...')
    xgbr = xgb.XGBRegressor(random_state = random_state)
    xgbr.fit(x_train, y_train['quantified_trend'])

    #Get predictions, calculate some metrics, return predictions dataframe
    if verbose: print('Getting predictions...')
    y_pred = xgbr.predict(x_test)
    if verbose: print('\nPERFORMANCE:')
    if verbose: print('R2 Score: {}'.format(r2_score(y_test['quantified_trend'], y_pred)))
    if verbose: print('Mean Squared Error: {}'.format(mean_squared_error(y_test['quantified_trend'], y_pred)))
    y_pred_df = pd.DataFrame(y_pred, columns = ['predictions'])

    #Join back all the testing data together 
    if verbose: print('\nJoining data back together...')
    test_dates_df = test_dates_df.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_pred_df = y_pred_df.reset_index(drop=True)
    final_test_set = pd.concat([test_dates_df, x_test, y_test, y_pred_df], axis=1)
    if verbose: print('Prediction dataframe returned. XGB finished.')
    return final_test_set



def generate_trade_signal(stock_df: pd.DataFrame, quantified_trend_col: str, strategy: str, quantile_value: float, verbose=True): 
    '''
    Generate the final trade signal used to create buy and sell decisions - set the buy and sell cutoff based on mean, 
    median, or quantile. 
    Args: 
        stock_df(pd.DataFrame): dataframe of stock data
        quantified_trend_col(str): name of quantified trend column
        strategy(str): mean, median, or quantile to set cutoff threshold between buy and sell (IMPORTANT PARAMETER - the 
        whole model and performance will be determined by this...)
        quantile_value(str): set regardless, only but used if your strategy is quantile_value
    Rtypes: 
        series(pd.Series): new column with the final trade signals generated 
    '''
    if verbose: print('\nGenerating trade signal (Up or Down)...')
    if strategy == 'mean': 
        cutoff_value = stock_df[quantified_trend_col].mean()
    if strategy == 'median': 
        cutoff_value = stock_df[quantified_trend_col].median()
    if strategy == 'quantile': 
        cutoff_value = stock_df[quantified_trend_col].quantile(quantile_value)

    stock_df['trade_signal'] = ''

    for row in range(0, len(stock_df)): 
        
        quantified_trend = stock_df[quantified_trend_col].loc[row]
        
        if quantified_trend > cutoff_value: 
            trend = 'Up'
        else: 
            trend = 'Down'
        
        stock_df['trade_signal'].loc[row] = trend

    series = stock_df['trade_signal']
    stock_df = stock_df.drop(columns='trade_signal', inplace=True)
    if verbose: print('Trade signal generated.')
    return series 


def generate_buy_decision(stock_df: pd.DataFrame, trade_signal_col_name: str, verbose=True): 
    '''
    Generate the buy and sell decisions in the dataframe based on the up and down condition in the trade signals column
    Args: 
        stock_df(pd.Dataframe) df to create buy and sell decisions on 
        trade_signal_col_name(str): name of column with trade signals created
    Rtypes: 
        series(pd.Series): column with the buy and sell decisions in it 
    '''
    if verbose: print('\nGenerating buy or sell decisions...')
    stock_df['buy_decision'] = ''

    for row in range(0, len(stock_df)): 
        
        current_condition = stock_df[trade_signal_col_name].iloc[row]
        prior_condition = stock_df[trade_signal_col_name].iloc[row-1]

        if (current_condition != prior_condition) & (prior_condition == 'Up'):
            final_decision = 'sell'
        elif (current_condition != prior_condition) & (prior_condition == 'Down'):
            final_decision = 'buy'
        else: 
            final_decision = 'hold'
        
        stock_df['buy_decision'].iloc[row] = final_decision

    series = stock_df['buy_decision']
    stock_df = stock_df.drop(columns=['buy_decision'], inplace=True)
    if verbose: print('Buy or sell decisions generated.')
    return series 


def calculate_profit_and_loss(stock_df: pd.DataFrame,  close_col: str, buy_decision_col: str, initial_cash: int, verbose=True):
    '''
    Make trading decisions, calculate profit and loss for a long only fund 
    Args:
        stock_df(pd.Dataframe): dataframe with  
        close_col(str): name of close column
        buy_decision_col(str): name of buy decision column
        initial_cash(int): starting cash position
    Rtypes: 
        stock_df(pd.Dataframe): Dataframe with new profit and loss columns added

    '''
    if verbose: print('\nMaking trading decisions...calculating profit and loss...')
    initial_close_price = stock_df[close_col].loc[0]
    shares = math.floor(initial_cash / initial_close_price)
    share_value = shares * initial_close_price
    leftover_cash = initial_cash - (shares * initial_close_price)

    stock_df['shares_owned'] = ''
    stock_df['total_value_of_shares'] = ''
    stock_df['remaining_cash'] = ''
    stock_df['total_portfolio_value'] = ''

    for i in range(0, len(stock_df)): 
        current_decision = stock_df[buy_decision_col].loc[i]
        todays_close_price = stock_df[close_col].loc[i]
        if (current_decision == 'buy'): 
            if i == 0:
                shares_to_buy = math.floor(leftover_cash / todays_close_price)
                shares = shares + shares_to_buy
                leftover_cash = leftover_cash - (shares_to_buy * todays_close_price)
                stock_df['shares_owned'].loc[i] = shares
                stock_df['total_value_of_shares'].loc[i] = shares * todays_close_price
                stock_df['remaining_cash'].loc[i] = leftover_cash
                stock_df['total_portfolio_value'].loc[i] = (shares * todays_close_price) + leftover_cash          
            else: 
                shares_to_buy = math.floor(stock_df['remaining_cash'].loc[i-1] / todays_close_price)
                shares = stock_df['shares_owned'].loc[i-1] + shares_to_buy
                stock_df['shares_owned'].loc[i] = shares
                stock_df['total_value_of_shares'].loc[i] = shares * todays_close_price
                stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1] - (shares_to_buy * todays_close_price)
                stock_df['total_portfolio_value'].loc[i] = stock_df['total_portfolio_value'].loc[i-1]
        if (current_decision == 'hold'): 
                if i == 0:    
                    stock_df['shares_owned'].loc[i] = shares
                    stock_df['remaining_cash'].loc[i] = leftover_cash
                else:  
                    stock_df['shares_owned'].loc[i] = stock_df['shares_owned'].loc[i-1]
                    stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1]
                stock_df['total_value_of_shares'].loc[i] = stock_df['shares_owned'].loc[i] * todays_close_price
                stock_df['total_portfolio_value'].loc[i] = stock_df['total_value_of_shares'].loc[i] + stock_df['remaining_cash'].loc[i]
        if (current_decision == 'sell'): 
                if i == 0: 
                    shares_to_sell = shares
                    stock_df['shares_owned'].loc[i] = 0
                    stock_df['total_value_of_shares'].loc[i] = 0 
                    stock_df['remaining_cash'].loc[i] = leftover_cash + (shares_to_sell * stock_df[close_col].loc[i])
                    stock_df['total_portfolio_value'].loc[i] = stock_df['remaining_cash'].loc[i]
                else: 
                    shares_to_sell = stock_df['shares_owned'].loc[i-1]
                    stock_df['shares_owned'].loc[i] = 0
                    stock_df['total_value_of_shares'].loc[i] = 0 
                    stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1] + (shares_to_sell * stock_df[close_col].loc[i])
                    stock_df['total_portfolio_value'].loc[i] = stock_df['remaining_cash'].loc[i]    
    
    if verbose: print('Trading decisions made.')
    starting_asset_value = stock_df[close_col].loc[0]
    ending_asset_value = stock_df[close_col].loc[len(stock_df) - 1]
    baseline_p_and_l =  round(((ending_asset_value - starting_asset_value) / starting_asset_value) * 100, 2)
    starting_portfolio_value = stock_df['total_portfolio_value'].loc[0]
    ending_portfolio_value = stock_df['total_portfolio_value'].loc[len(stock_df) - 1]
    strategy_p_and_l = round(((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100, 2)
    alpha = round(strategy_p_and_l - baseline_p_and_l, 2)
    if verbose: print('\nStarting Value Asset: {}'.format(starting_asset_value))
    if verbose: print('Ending Value Asset: {}'.format(ending_asset_value))
    if verbose: print('Baseline P&L: {}%'.format(baseline_p_and_l))
    if verbose: print('\nStarting Portfolio Value: {}'.format(starting_portfolio_value))
    if verbose: print('Ending Portfolio Value: {}'.format(ending_portfolio_value))
    if verbose: print('STRATEGY P&L: {}%'.format(strategy_p_and_l))
    if verbose: print('ALPHA: {}%'.format(alpha))

    return stock_df


def view_portfolio_df(stock_df: pd.DataFrame, total_portfolio_value_col: str):
    '''
    View the portfolio value over time in a graph
    Args: 
        stock_df(pd.DataFrame): Stock df with portfolio value in it
        total_portfolio_value_col(str): Name of column to graph
    Rtypes: 
        None
    '''
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[total_portfolio_value_col] ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.set_title(ticker,fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Porfolio Value' , fontsize=18)
    # legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    return None



def view_stock_with_decision(stock_df: pd.DataFrame, ticker: str, close_col: str, buy_decision_col: str):
    '''
    View stock dataframe with trading decisions in it 
    Args: 
        stock_dfD(pd.Dataframe): stock df with buy and sell decisions
        ticker(str): ticker name string
        close_col(str): close column name string
        buy_decision_col(str): buy decision column name 
    Rtypes: 
        None
    '''


    def create_buy_column(row): 
        if row[buy_decision_col] == 'buy': 
            return row[close_col]
        else: 
            None

    def create_sell_column(row): 
        if row[buy_decision_col] == 'sell': 
            return row[close_col]
        else: 
            return None 
    stock_df['sell_close'] = stock_df.apply(create_sell_column, axis=1)
    stock_df['buy_close'] = stock_df.apply(create_buy_column, axis=1)
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[close_col] , label = 'Close' ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.scatter(stock_df.index , stock_df['buy_close'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax.scatter(stock_df.index , stock_df['sell_close'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax.set_title(ticker + " Price History with Buy and Sell Signals",fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Close Prices' , fontsize=18)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    stock_df = stock_df.drop(columns=['sell_close', 'buy_close'])
    return None




# asset_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD',\
#     'MATIC-USD', 'AVAX-USD', 'TRX-USD', 'UNI1-USD', 'LTC-USD', 'CRO-USD', 'ATOM-USD', 'XMR-USD', 'XLM-USD']


# ticker ='BTC-USD'
# print('\n' + ticker)
# historical_data_df = get_stock_data(stock = ticker, period='5Y', verbose= False)
# # historical_data_df = get_stock_data(stock = current_ticker, period='10Y', verbose= False)

# historical_data_df['sma_15'] = simple_moving_average(stock_df = historical_data_df, close_col='Close',\
#     window=15, fillna=True, verbose=False)

# historical_data_df['trend_analysis'] = trend_analysis(stock_df= historical_data_df, window= 5, sma_col= 'sma_15',\
#     close_col= 'Close', verbose=False)

# historical_data_df['quantified_trend'] = get_quantified_trend(stock_df= historical_data_df, close_col= 'Close',\
#     trend_analysis_col= 'trend_analysis', window=3, verbose=False)

# # print('\nCreating technical features...')
# historical_data_df = ta.add_all_ta_features(historical_data_df, open="Open", high="High", low="Low",\
#         close = "Close", volume="Volume", fillna=True)

# test_df = train_xgb(stock_df = historical_data_df, train_set_percent= 0.7, random_state = 1, verbose=True)

# test_df['trade_signal'] = generate_trade_signal(stock_df = test_df, quantified_trend_col = 'predictions', strategy= 'quantile', quantile_value= 0.80, verbose=False)

# test_df['buy_decision'] = generate_buy_decision(stock_df = test_df, trade_signal_col_name= 'trade_signal', verbose=False)

# earnings_df = test_df[['Date', 'Close', 'quantified_trend', 'predictions', 'trade_signal', 'buy_decision']]

# final_stock_df = calculate_profit_and_loss(stock_df= earnings_df,  close_col= 'Close', buy_decision_col= 'buy_decision', initial_cash=100000, verbose=True)

# view_portfolio_df(stock_df= final_stock_df, total_portfolio_value_col= 'total_portfolio_value')

# view_stock_with_decision(stock_df= earnings_df, ticker= ticker, close_col= 'Close', buy_decision_col= 'buy_decision')





tuning_df = pd.DataFrame(columns=['ticker', 'sma_window', 'trend_analysis_window', 'quantified_trend_window', 'quantile', 'starting_asset_value',\
    'ending_asset_value', 'baseline_p_and_l', 'starting_portfolio_value', 'ending_portfolio_value', 'strategy_p_and_l', 'alpha'])

sma_window_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
trend_analysis_window_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
quantified_trend_window_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
quantile_value_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

iterator = 0
total_iterations = len(sma_window_list) * len(trend_analysis_window_list) * len(quantified_trend_window_list) * len(quantile_value_list)


# sma_window_list = [12, 15]
# trend_analysis_window_list = [12, 15]
# quantified_trend_window_list = [12, 15]
# quantile_value_list = [0.65, 0.75]


for sma_window_counter in range(0, len(sma_window_list)): 

    sma_window = sma_window_list[sma_window_counter]

    for trend_analysis_window_counter in range(0, len(trend_analysis_window_list)): 

        trend_analysis_window = trend_analysis_window_list[trend_analysis_window_counter]

        for quantified_trend_counter in range(0, len(quantified_trend_window_list)): 

            quantified_trend_window = quantified_trend_window_list[quantified_trend_counter]

            for quantile_counter in range(0, len(quantile_value_list)): 

                quantile = quantile_value_list[quantile_counter]
                
                ticker ='BTC-USD'
                print('\n' + ticker)
                historical_data_df = get_stock_data(stock = ticker, period='5Y', verbose= False)
                # historical_data_df = get_stock_data(stock = current_ticker, period='10Y', verbose= False)

                historical_data_df['sma_15'] = simple_moving_average(stock_df = historical_data_df, close_col='Close',\
                    window=sma_window, fillna=True, verbose=False)

                historical_data_df['trend_analysis'] = trend_analysis(stock_df= historical_data_df, window= trend_analysis_window, sma_col= 'sma_15',\
                    close_col= 'Close', verbose=False)

                historical_data_df['quantified_trend'] = get_quantified_trend(stock_df= historical_data_df, close_col= 'Close',\
                    trend_analysis_col= 'trend_analysis', window=quantified_trend_window, verbose=False)

                # print('\nCreating technical features...')
                historical_data_df = ta.add_all_ta_features(historical_data_df, open="Open", high="High", low="Low",\
                        close = "Close", volume="Volume", fillna=True)

                test_df = train_xgb(stock_df = historical_data_df, train_set_percent= 0.7, random_state = 1, verbose=True)

                test_df['trade_signal'] = generate_trade_signal(stock_df = test_df, quantified_trend_col = 'predictions', strategy= 'quantile', quantile_value= quantile, verbose=False)

                test_df['buy_decision'] = generate_buy_decision(stock_df = test_df, trade_signal_col_name= 'trade_signal', verbose=False)

                earnings_df = test_df[['Date', 'Close', 'quantified_trend', 'predictions', 'trade_signal', 'buy_decision']]

                final_stock_df = calculate_profit_and_loss(stock_df= earnings_df,  close_col= 'Close', buy_decision_col= 'buy_decision', initial_cash=100000, verbose=True)


                starting_asset_value = final_stock_df['Close'].loc[0]
                ending_asset_value = final_stock_df['Close'].loc[len(final_stock_df) - 1]
                baseline_p_and_l =  round(((ending_asset_value - starting_asset_value) / starting_asset_value) * 100, 2)
                starting_portfolio_value = final_stock_df['total_portfolio_value'].loc[0]
                ending_portfolio_value = final_stock_df['total_portfolio_value'].loc[len(final_stock_df) - 1]
                strategy_p_and_l = round(((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100, 2)
                alpha = round(strategy_p_and_l - baseline_p_and_l, 2)

                temp_df = {'ticker': ticker, 'sma_window': sma_window, 'trend_analysis_window': trend_analysis_window, 'quantified_trend_window': quantified_trend_window,\
                    'quantile': quantile, 'starting_asset_value': starting_asset_value, 'ending_asset_value': ending_asset_value, 'baseline_p_and_l': baseline_p_and_l,\
                        'starting_portfolio_value': starting_portfolio_value, 'ending_portfolio_value': ending_portfolio_value, 'strategy_p_and_l': strategy_p_and_l,\
                            'alpha': alpha}
                
                tuning_df = tuning_df.append(temp_df, ignore_index=True)
                print(tuning_df)

                iterator = iterator + 1
                print('\nFinished round ' + str(iterator) + ' of ' + str(total_iterations))

tuning_df.to_csv('tuning_df.csv')




'''
TO DO: 
- Add new trading algo with fractional shares and commission

FUTURE: 
- Leverage? 
'''