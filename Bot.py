import websocket
import time
import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import krakenex
import threading

# Load environment variables from .env file
load_dotenv()

# Fetch API key and secret from environment variables
API_KEY = os.getenv('KRACKEN_API_KEY_DEMO')
API_SECRET = os.getenv('KRACKEN_API_SECRET_DEMO')

# Define the pair
PAIR = os.getenv('PAIR', 'XBT/USD')  # Bitcoin to USD

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize a DataFrame to store trade data
global_data = pd.DataFrame(columns=['time', 'price', 'volume'])

# Initialize Kraken client
kraken = krakenex.API()
kraken.key = API_KEY
kraken.secret = API_SECRET

# Global variable to track WebSocket connection status
global_websocket = None
ping_thread_running = False

# Function to handle incoming messages
def on_message(ws, message):
    try:
        message = json.loads(message)
        if isinstance(message, list) and len(message) == 5:
            trade_data_list = message[1]
            event_name = message[2]
            pair = message[3]
            if event_name == 'trade' and pair == PAIR:
                if isinstance(trade_data_list, list) and len(trade_data_list) > 0:
                    for trade_data in trade_data_list:
                        if isinstance(trade_data, list) and len(trade_data) >= 3:
                            trade_time = pd.to_datetime(float(trade_data[2]), unit='s')
                            trade_data_entry = {
                                'time': trade_time,
                                'price': float(trade_data[0]),
                                'volume': float(trade_data[1])
                            }
                            update_global_data(trade_data_entry)
                            logger.info(f"Received new trade data: {trade_data_entry}")
                            # Optionally, you can call backtest_strategy here if needed
                            # backtest_strategy(global_data)
                        else:
                            logger.info(f"Unhandled trade data entry: {trade_data}")
                else:
                    logger.info(f"Unhandled trade data list: {trade_data_list}")
            else:
                logger.info(f"Unhandled message: {message}")
        elif isinstance(message, dict):
            event = message.get('event')
            if event == 'heartbeat':
                logger.info("Received heartbeat message.")
                send_pong(ws)
            elif event == 'pong':
                logger.info("Received pong message.")
            elif event == 'systemStatus':
                logger.info(f"Received system status message: {message}")
            elif event == 'subscriptionStatus':
                if message.get('status') == 'subscribed':
                    logger.info(f"Subscribed to {PAIR} trade data.")
                else:
                    logger.error(f"Subscription error: {message}")
            else:
                logger.info(f"Unhandled message: {message}")
        else:
            logger.info(f"Unhandled message: {message}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

# Function to send pong message
def send_pong(ws):
    try:
        pong_message = {"event": "pong"}
        ws.send(json.dumps(pong_message))
        logger.info("Pong sent to WebSocket")
    except Exception as e:
        logger.error(f"Error sending pong: {e}")

# Function to handle WebSocket errors
def on_error(ws, error):
    global global_websocket
    global ping_thread_running
    logger.error(f"WebSocket error: {error}")
    reconnect_websocket()

# Function to handle WebSocket close
def on_close(ws, close_status_code, close_msg):
    global global_websocket
    global ping_thread_running
    global_websocket = None
    ping_thread_running = False
    logger.info(f"WebSocket closed with status code {close_status_code}: {close_msg}")
    reconnect_websocket()

# Function to handle WebSocket open
def on_open(ws):
    logger.info("WebSocket connection established.")
    subscribe_to_trade_data(ws)

# Function to subscribe to trade data
def subscribe_to_trade_data(ws):
    subscription_message = {
        "event": "subscribe",
        "subscription": {"name": "trade"},
        "pair": [PAIR]
    }
    ws.send(json.dumps(subscription_message))
    logger.info(f"Subscription message: {subscription_message}")
    logger.info("Subscription message sent.")

def update_global_data(trade_data):
    global global_data
    new_data = pd.DataFrame([trade_data])
    global_data = pd.concat([global_data, new_data], ignore_index=True)
    global_data.set_index('time', inplace=True)
    global_data = global_data.astype({
        'price': float,
        'volume': float
    })

def backtest_strategy(data):
    """
    Backtest a simple moving average crossover strategy with stop loss and take profit.
    :param data: DataFrame containing OHLC data
    """
    # Calculate moving averages
    data['SMA_50'] = data['price'].rolling(window=50).mean()
    data['SMA_200'] = data['price'].rolling(window=200).mean()

    data['position'] = 0
    data['entry_price'] = np.nan
    data['pnl'] = 0.0

    position = 0
    entry_price = None

    for i in range(50, len(data)):
        if pd.notnull(data['SMA_50'].iloc[i]) and pd.notnull(data['SMA_200'].iloc[i]):
            if position == 0:
                if data['SMA_50'].iloc[i] > data['SMA_200'].iloc[i]:
                    position = 1
                    entry_price = data['price'].iloc[i]
                    data.at[data.index[i], 'position'] = 1
                    data.at[data.index[i], 'entry_price'] = entry_price
                    logger.info(f"Long entry at {entry_price}")
                    execute_trade("buy")
                elif data['SMA_50'].iloc[i] < data['SMA_200'].iloc[i]:
                    position = -1
                    entry_price = data['price'].iloc[i]
                    data.at[data.index[i], 'position'] = -1
                    data.at[data.index[i], 'entry_price'] = entry_price
                    logger.info(f"Short entry at {entry_price}")
                    execute_trade("sell")
            elif position == 1:
                if data['price'].iloc[i] < entry_price * (1 - 0.02):
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = data['price'].iloc[i] - entry_price
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Long exit at {data['price'].iloc[i]} with PnL {pnl} (Stop Loss)")
                    execute_trade("sell")
                elif data['price'].iloc[i] > entry_price * (1 + 0.03):
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = data['price'].iloc[i] - entry_price
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Long exit at {data['price'].iloc[i]} with PnL {pnl} (Take Profit)")
                    execute_trade("sell")
                elif data['price'].iloc[i] < data['SMA_50'].iloc[i]:
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = data['price'].iloc[i] - entry_price
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Long exit at {data['price'].iloc[i]} with PnL {pnl}")
                    execute_trade("sell")
            elif position == -1:
                if data['price'].iloc[i] > entry_price * (1 + 0.02):
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = entry_price - data['price'].iloc[i]
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Short exit at {data['price'].iloc[i]} with PnL {pnl} (Stop Loss)")
                    execute_trade("buy")
                elif data['price'].iloc[i] < entry_price * (1 - 0.03):
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = entry_price - data['price'].iloc[i]
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Short exit at {data['price'].iloc[i]} with PnL {pnl} (Take Profit)")
                    execute_trade("buy")
                elif data['price'].iloc[i] > data['SMA_50'].iloc[i]:
                    position = 0
                    data.at[data.index[i], 'position'] = 0
                    pnl = entry_price - data['price'].iloc[i]
                    data.at[data.index[i], 'pnl'] = pnl
                    entry_price = None
                    logger.info(f"Short exit at {data['price'].iloc[i]} with PnL {pnl}")
                    execute_trade("buy")

    data['cumulative_pnl'] = data['pnl'].cumsum()

    if data['pnl'].std() != 0:
        annualized_return = (data['cumulative_pnl'].iloc[-1] / len(data)) * 365
        max_drawdown = (data['cumulative_pnl'].cummax() - data['cumulative_pnl']).max()
        sharpe_ratio = data['pnl'].mean() / data['pnl'].std() * np.sqrt(365)
        logger.info(f"Annualized Return: {annualized_return}")
        logger.info(f"Maximum Drawdown: {max_drawdown}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio}")
    else:
        logger.info('Annualized Return: nan')
        logger.info('Maximum Drawdown: nan')
        logger.info('Sharpe Ratio: nan')

    plt.figure(figsize=(14, 7))
    plt.plot(data['cumulative_pnl'], label='Cumulative PnL')
    plt.title('Cumulative Profit and Loss (PnL)')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    plt.show()

def execute_trade(action):
    """
    Execute a trade on Kraken.
    :param action: 'buy' or 'sell'
    """
    try:
        volume = '0.001'  # Volume in XBT
        if action == "buy":
            response = kraken.query_private('AddOrder', {
                'pair': PAIR,
                'type': 'buy',
                'ordertype': 'market',
                'volume': volume
            })
            logger.info(f"Buy order placed: {response}")
        elif action == "sell":
            response = kraken.query_private('AddOrder', {
                'pair': PAIR,
                'type': 'sell',
                'ordertype': 'market',
                'volume': volume
            })
            logger.info(f"Sell order placed: {response}")
    except Exception as e:
        logger.error(f"Error executing trade: {e}")

def send_ping():
    global global_websocket
    global ping_thread_running
    while ping_thread_running:
        try:
            if global_websocket and global_websocket.sock:
                ping_message = {"event": "ping"}
                global_websocket.send(json.dumps(ping_message))
                logger.info("Ping sent to WebSocket")
            else:
                logger.error("WebSocket connection is closed. Stopping ping thread.")
                ping_thread_running = False
        except Exception as e:
            logger.error(f"Error sending ping: {e}")
            ping_thread_running = False
        time.sleep(30)

def start_websocket():
    global global_websocket
    global ping_thread_running
    websocket.enableTrace(False)
    global_websocket = websocket.WebSocketApp(
        "wss://ws.kraken.com",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    global_websocket.on_open = on_open

    logger.info("Connecting to WebSocket...")
    ping_thread = threading.Thread(target=send_ping)
    ping_thread.daemon = True
    ping_thread.start()
    ping_thread_running = True

    try:
        global_websocket.run_forever()
    except Exception as e:
        logger.error(f"WebSocket run_forever error: {e}")

def reconnect_websocket():
    global ping_thread_running
    ping_thread_running = False
    time.sleep(5)
    start_websocket()

# Main function
def main():
    logger.info(f"PAIR variable set to: {PAIR}")
    start_websocket()

if __name__ == "__main__":
    main()