from matplotlib import pyplot as plt
from datetime import datetime


with open("logs.txt", "r") as file:
    data, sessions, sessions_data = [], [], {}
    for i in file.readlines():
        data.append({d.split("=")[0]: d.split("=")[1] for d in [s for s in i.replace("\n", "").split("/")]})
for i in data:
    sessions.append(i['session_id']) if i['session_id'] not in sessions else ...
for session in sessions:
    sessions_data[session] = {"order_data": [], 'transaction_prices': [], 'buy_side_transaction_prices': [],
                              'sell_side_transaction_prices': [], "data": [], "candles": [], "last_prices": [],
                              "signals": [], "open_candles": [], "high_candles": [], "low_candles": [],
                              "base_asset_balances": [], "quote_asset_balances": [], "base_asset_quantity": [],
                              "quote_asset_quantity": [], "order_signals": [], "transaction_times": [],
                              "close_candles": [], "buy_side_quote_asset_balances": [],
                              "sell_side_quote_asset_balances": [], "datetime": [],
                              "buy_side_transaction_datetime": [], "sell_side_transaction_datetime": [],
                              "tp_sell_side_transaction_prices": [], "tp_sell_side_quote_asset_balances": [],
                              "tp_sell_side_transaction_datetime": [],
                              "tp_order_data": [], "tp_transaction_prices": [],
                              "tp_quote_asset_balances": [], "tp_base_asset_balances": [],
                              "tp_base_asset_quantity": [], "tp_quote_asset_quantity": [],
                              "tp_order_signals": [], "tp_transaction_times": [],
                              "sl_sell_side_transaction_prices": [], "sl_sell_side_quote_asset_balances": [],
                              "sl_sell_side_transaction_datetime": [],
                              "sl_order_data": [], "sl_transaction_prices": [],
                              "sl_quote_asset_balances": [], "sl_base_asset_balances": [],
                              "sl_base_asset_quantity": [], "sl_quote_asset_quantity": [],
                              "sl_order_signals": [], "sl_transaction_times": [],}
    for i in data:
        if i['session_id'] == session:
            if i['order_response'] == 'True':
                if i['side'] == 'BUY':
                    sessions_data[session]['buy_side_transaction_prices'].append(float(i['transaction_price']))
                    sessions_data[session]['buy_side_quote_asset_balances'].append(float(i['quote_asset_balance']))
                    sessions_data[session]['buy_side_transaction_datetime'].append(
                        datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
                if i['side'] == 'SELL':
                    sessions_data[session]['sell_side_transaction_prices'].append(float(i['transaction_price']))
                    sessions_data[session]['sell_side_quote_asset_balances'].append(float(i['quote_asset_balance']))
                    sessions_data[session]['sell_side_transaction_datetime'].append(
                        datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
                sessions_data[session]["order_data"].append({i1: i[i1] for i1 in i if i1 not in ('session_id',)})
                sessions_data[session]["transaction_prices"].append(float(i['transaction_price']))
                sessions_data[session]["quote_asset_balances"].append(float(i['quote_asset_balance']))
                sessions_data[session]["base_asset_balances"].append(float(i['base_asset_balance']))
                sessions_data[session]["base_asset_quantity"].append(float(i['base_asset_quantity']))
                sessions_data[session]["quote_asset_quantity"].append(float(i['quote_asset_quantity']))
                sessions_data[session]["order_signals"].append(int(i['signal']))
                sessions_data[session]["transaction_times"].append(
                    datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
            if i['take_profit_order_response'] == 'True':
                if i['side'] == 'SELL':
                    sessions_data[session]['tp_sell_side_transaction_prices'].append(float(i['transaction_price']))
                    sessions_data[session]['tp_sell_side_quote_asset_balances'].append(float(i['quote_asset_balance']))
                    sessions_data[session]['tp_sell_side_transaction_datetime'].append(
                        datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
                sessions_data[session]["tp_order_data"].append({i1: i[i1] for i1 in i if i1 not in ('session_id',)})
                sessions_data[session]["tp_transaction_prices"].append(float(i['transaction_price']))
                sessions_data[session]["tp_quote_asset_balances"].append(float(i['quote_asset_balance']))
                sessions_data[session]["tp_base_asset_balances"].append(float(i['base_asset_balance']))
                sessions_data[session]["tp_base_asset_quantity"].append(float(i['base_asset_quantity']))
                sessions_data[session]["tp_quote_asset_quantity"].append(float(i['quote_asset_quantity']))
                sessions_data[session]["tp_order_signals"].append(int(i['signal']))
                sessions_data[session]["tp_transaction_times"].append(
                    datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
            if i['stop_loss_order_response'] == 'True':
                if i['side'] == 'SELL':
                    sessions_data[session]['sl_sell_side_transaction_prices'].append(float(i['transaction_price']))
                    sessions_data[session]['sl_sell_side_quote_asset_balances'].append(float(i['quote_asset_balance']))
                    sessions_data[session]['sl_sell_side_transaction_datetime'].append(
                        datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
                sessions_data[session]["sl_order_data"].append({i1: i[i1] for i1 in i if i1 not in ('session_id',)})
                sessions_data[session]["sl_transaction_prices"].append(float(i['transaction_price']))
                sessions_data[session]["sl_quote_asset_balances"].append(float(i['quote_asset_balance']))
                sessions_data[session]["sl_base_asset_balances"].append(float(i['base_asset_balance']))
                sessions_data[session]["sl_base_asset_quantity"].append(float(i['base_asset_quantity']))
                sessions_data[session]["sl_quote_asset_quantity"].append(float(i['quote_asset_quantity']))
                sessions_data[session]["sl_order_signals"].append(int(i['signal']))
                sessions_data[session]["sl_transaction_times"].append(
                    datetime.strptime(i['transaction_time'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
            elif i['order_response'] == 'False' and i['take_profit_order_response'] == 'False' \
                    and i['stop_loss_order_response'] == 'False':
                candle_data = {}
                for s in i['candle'].split(","):
                    candle_data[s.split(":")[0]] = float(s.split(":")[1])
                i['candle'] = candle_data
                sessions_data[session]["data"].append({i1: i[i1] for i1 in i if i1 not in ('session_id',)})
                sessions_data[session]["candles"].append(candle_data)
                sessions_data[session]["last_prices"].append(float(i['last_price']))
                sessions_data[session]["signals"].append(int(i['signal']))
                sessions_data[session]["open_candles"].append(float(i['candle']["open"]))
                sessions_data[session]["high_candles"].append(float(i['candle']["high"]))
                sessions_data[session]["low_candles"].append(float(i['candle']["low"]))
                sessions_data[session]["close_candles"].append(float(i['candle']["close"]))
                sessions_data[session]["datetime"].append(
                    datetime.strptime(i['datetime'].split(".")[0], '%Y-%m-%d %H:%M:%S'))
print("Сессии:")
for n, session in enumerate(sessions):
    print(f"{n}. {session} real_order={sessions_data[session]['data'][-1]['real_order']} "
          f"{sessions_data[session]['data'][0]['start_datetime']} "
          f"({sessions_data[session]['data'][-1]['working_sec']} sec)")
session = sessions[int(input("Укажите номер сессии:"))]
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [2, 0.75]})
ax1.plot(sessions_data[session]['datetime'], sessions_data[session]['close_candles'], color="#ffffff", label='close')
ax1.plot(sessions_data[session]['datetime'], sessions_data[session]['open_candles'],
         color="#dfff4f", alpha=0.2, label='open')
ax1.plot(sessions_data[session]['datetime'], sessions_data[session]['high_candles'],
         color="#4fff61", alpha=0.2, label='high')
ax1.plot(sessions_data[session]['datetime'], sessions_data[session]['low_candles'],
         color="#ff4f4f", alpha=0.2, label='low')
ax1.plot(sessions_data[session]['datetime'], sessions_data[session]['last_prices'],
         color="#ffffff", alpha=0.1, label='last prices')
ax1.plot(sessions_data[session]['buy_side_transaction_datetime'],
         sessions_data[session]['buy_side_transaction_prices'], "^", color="#38ff84", label='buy')
ax1.plot(sessions_data[session]['sell_side_transaction_datetime'],
         sessions_data[session]['sell_side_transaction_prices'], "v", color="#ff384c", label='sell')
ax1.plot(sessions_data[session]['tp_sell_side_transaction_datetime'],
         sessions_data[session]['tp_sell_side_transaction_prices'], ".", color="#339cff", label='take profit')
ax1.plot(sessions_data[session]['sl_sell_side_transaction_datetime'],
         sessions_data[session]['sl_sell_side_transaction_prices'], ".", color="#ffb833", label='stop loss')
ax1.legend(loc='upper left', frameon=False, prop={"size": 8})
ax2.plot([i for i in range(len(sessions_data[session]['sell_side_quote_asset_balances']))],
         [i * 100 / sessions_data[session]['sell_side_quote_asset_balances'][0]
         for i in sessions_data[session]['sell_side_quote_asset_balances']], color="#4fff92", label='% balances')
ax2.legend(loc='upper left', frameon=False, prop={"size": 8})
plt.show()
