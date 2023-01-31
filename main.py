from datetime import datetime
from binance import Client as BinanceClient
from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager
import keys
import settings
S = settings


class MyCustomError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return 'MyCustomError, {0} '.format(self.message) if self.message else 'MyCustomError has been raised'


class BinanceCandleInterval:
    CANDLE_INTERVAL_1_MIN = BinanceClient.KLINE_INTERVAL_1MINUTE
    CANDLE_INTERVAL_5_MIN = BinanceClient.KLINE_INTERVAL_5MINUTE
    CANDLE_INTERVAL_15_MIN = BinanceClient.KLINE_INTERVAL_15MINUTE
    CANDLE_INTERVAL_HOUR = BinanceClient.KLINE_INTERVAL_1HOUR
    CANDLE_INTERVAL_DAY = BinanceClient.KLINE_INTERVAL_1DAY


class Sides:
    BUY = 'BUY'
    SELL = 'SELL'


class CandleType:
    ClassicCandles: str = "cl"
    HeikinAshiCandles: str = "ha"


class StreamChannels:
    Candle_1M = "candle_1m"
    Candle_5M = "candle_5m"
    Candle_15M = "candle_15m"
    CandleHour = "candle_1h"
    CandleDay = "candle_1d"
    LastPrice = "last_price"


class Tools:
    @staticmethod
    def milliseconds_to_datatime(milliseconds: int):
        return datetime.fromtimestamp(milliseconds/1000)

    @staticmethod
    def loger(data, max_lines: int = 10000, del_lines: int = 100, file_name: str = "logs.txt"):
        try:
            with open(file_name, "r+") as file:
                len_file = len(file.readlines())
                file.write(data + f"/datetime={datetime.now()}\n")
        except FileNotFoundError:
            with open(file_name, "w") as file:
                len_file = 0
                file.write(data + f"/datetime={datetime.now()}\n")
        if len_file >= max_lines:
            with open(file_name, 'r') as file:
                read_file = "".join(i for i in file.readlines()[del_lines:])
            with open(file_name, 'w') as file:
                file.write(read_file)


def ema_smoothing(values, period: int, w=1.0):
    ema_results, alpha = [sum(values[:period]) / period], w / (period + w - 1)
    for value in values[period:]:
        ema_results.append(alpha * value + (1 - alpha) * ema_results[-1])
    return ema_results


def ma_smoothing(values, period: int):
    return [sum(values[n:n + period]) / period for n in range(len(values) - period + 1)]


class CandleHandler:
    class CandleArguments:
        def __init__(self, open_, high_, low_, close_, volume_, datetime_):
            self.open = open_
            self.high = high_
            self.low = low_
            self.close = close_
            self.volume = volume_
            self.datetime = datetime_

        def __str__(self):
            return f"{self.__dict__}"

    def __init__(self, candle_data):
        try:
            self.candle = CandleHandler.CandleArguments(
                open_=float(candle_data[1]),
                high_=float(candle_data[2]),
                low_=float(candle_data[3]),
                close_=float(candle_data[4]),
                volume_=float(candle_data[5]),
                datetime_=Tools.milliseconds_to_datatime(int(candle_data[0]))
            )
        except KeyError:
            self.candle = CandleHandler.CandleArguments(
                open_=float(candle_data['open_price']),
                high_=float(candle_data['high_price']),
                low_=float(candle_data['low_price']),
                close_=float(candle_data['close_price']),
                volume_=float(candle_data['base_volume']),
                datetime_=Tools.milliseconds_to_datatime(int(candle_data['kline_close_time']))
            )

    def ha_candle(self, previous_candle: CandleArguments):
        self.candle = CandleHandler.CandleArguments(
            open_=(previous_candle.open + previous_candle.close) / 2,
            high_=max((previous_candle.open + previous_candle.close) / 2, self.candle.high,
                      (self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4),
            low_=min((previous_candle.open + previous_candle.close) / 2, self.candle.low,
                     (self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4),
            close_=(self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4,
            volume_=self.candle.volume,
            datetime_=self.candle.datetime,
        ) if previous_candle else CandleHandler.CandleArguments(
            open_=(self.candle.open + self.candle.close) / 2,
            high_=self.candle.high,
            low_=self.candle.low,
            close_=(self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4,
            volume_=self.candle.volume,
            datetime_=self.candle.datetime,
        )
        return self.candle


class Indicators:
    class ChandelierExitArguments:
        def __init__(self, chandelier_exit: tuple):
            self.chandelier_exit_long = chandelier_exit[0]
            self.chandelier_exit_short = chandelier_exit[1]
            self.last_chandelier_exit_long = chandelier_exit[0][-1]
            self.last_chandelier_exit_short = chandelier_exit[1][-1]
            self.prev_chandelier_exit_long = chandelier_exit[0][-2]
            self.prev_chandelier_exit_short = chandelier_exit[1][-2]

    class AdxArguments:
        def __init__(self, adx: tuple):
            self.adx = adx[0]
            self.di_plus = adx[1]
            self.di_minus = adx[2]
            self.last_adx = adx[0][-1]
            self.last_di_plus = adx[1][-1]
            self.last_di_minus = adx[2][-1]
            self.prev_adx = adx[0][-2]
            self.prev_di_plus = adx[1][-2]
            self.prev_di_minus = adx[2][-2]

    def __init__(self, candles: dict):
        self.candles = candles
        self.chandelier_exit = None
        self.adx = None

    def chandelier_exit_calc(self, period: int, factor: float, w: float = 1.0):
        atr = ema_smoothing([max(self.candles["high"][n] - self.candles["low"][n],
                                 abs(self.candles["high"][n] - self.candles["close"][n - 1 if n > 0 else 0]),
                                 abs(self.candles["low"][n] - self.candles["close"][n - 1 if n > 0 else 0]))
                             for n in range(len(self.candles["close"]))], period, w)
        chandelier_exit_long = [max(self.candles["high"][:period]) - atr[0] * factor]
        chandelier_exit_short = [min(self.candles["low"][:period]) + atr[0] * factor]
        for n in range(1, len(self.candles["close"]) - period + 1):
            chandelier_exit_long.append(
                max(max(self.candles["high"][n:n + period]) - atr[n] * factor, chandelier_exit_long[-1])
                if self.candles["close"][n + period - 2] > chandelier_exit_long[-1]
                else max(self.candles["high"][n:n + period]) - atr[n] * factor)
            chandelier_exit_short.append(
                min(min(self.candles["low"][n:n + period]) + atr[n] * factor, chandelier_exit_short[-1])
                if self.candles["close"][n + period - 2] < chandelier_exit_short[-1]
                else min(self.candles["low"][n:n + period]) + atr[n] * factor)
        self.chandelier_exit = Indicators.ChandelierExitArguments((chandelier_exit_long, chandelier_exit_short))
        return self.chandelier_exit

    def adx_calc(self, period: int, w: float = 1.0):
        tr, dm_plus, dm_minus, di_plus, di_minus, dx = [], [], [], [], [], []
        for n in range(len(self.candles["close"])):
            pn = n - 1 if n > 0 else 0
            tr.append(max(self.candles["high"][n] - self.candles["low"][n],
                          abs(self.candles["high"][n] - self.candles["close"][pn]),
                          abs(self.candles["low"][n] - self.candles["close"][pn])))
            high_dif, low_dif = self.candles["high"][n] - self.candles["high"][pn]
            low_dif = self.candles["low"][pn] - self.candles["low"][n]
            dm_plus.append(high_dif if high_dif > low_dif and high_dif > 0 else 0)
            dm_minus.append(low_dif if low_dif > high_dif and low_dif > 0 else 0)
        atr = ema_smoothing(tr, period, w)
        smoothed_dm_plus, smoothed_dm_minus = ema_smoothing(dm_plus, period, w), ema_smoothing(dm_minus, period, w)
        for n in range(len(atr)):
            try:
                di_plus.append(smoothed_dm_plus[n] / atr[n] * 100), di_minus.append(smoothed_dm_minus[n] / atr[n] * 100)
            except ZeroDivisionError:
                di_plus.append(0), di_minus.append(0)
            dx.append(abs(di_plus[-1] - di_minus[-1]) / abs(di_plus[-1] + di_minus[-1]) * 100
                      if di_plus[-1] + di_minus[-1] > 0 else 0)
        self.adx = Indicators.AdxArguments((ema_smoothing(dx, period, w), di_plus, di_minus))
        return self.adx


class CandleServices:
    def __init__(self, candles: list[CandleHandler.CandleArguments], only_methods: bool = False):
        self.candles = candles
        if not only_methods:
            self.candles_quantity = len(candles)
            self.last_open_candle = candles[-1].open
            self.last_high_candle = candles[-1].high
            self.last_low_candle = candles[-1].low
            self.last_close_candle = candles[-1].close
            self.last_volume_candle = candles[-1].volume
            self.last_datatime_candle = candles[-1].datetime
            self.previous_open_candle = candles[-2].open
            self.previous_high_candle = candles[-2].high
            self.previous_low_candle = candles[-2].low
            self.previous_close_candle = candles[-2].close
            self.last_close_candle_positivity_negativity = \
                "positive" if self.last_close_candle >= candles[-2].close else "negative"

    def candles_dict(self):
        candles_dict = {'open': (), 'high': (), 'low': (), 'close': (), 'volume': (), 'datetime': ()}
        for candle in self.candles:
            candles_dict['open'] += (candle.open,)
            candles_dict['high'] += (candle.high,)
            candles_dict['low'] += (candle.low,)
            candles_dict['close'] += (candle.close,)
            candles_dict['volume'] += (candle.volume,)
            candles_dict['datetime'] += (candle.datetime,)
        return candles_dict

    @staticmethod
    def candles_dict_slice(candles_dict: dict, start: int, stop: int = None):
        return {'open': candles_dict['open'][start:stop], 'high': candles_dict['high'][start:stop],
                'low': candles_dict['low'][start:stop], 'close': candles_dict['close'][start:stop],
                'volume': candles_dict['volume'][start:stop], 'datetime': candles_dict['datetime'][start:stop]}


class Strategies:
    def __init__(self, signal: int, indicators: Indicators):
        self.signal = signal
        self.indicators = indicators

    def chandelier_exit_strategy(self):
        candles = self.indicators.candles
        chandelier_exit = self.indicators.chandelier_exit
        if candles["close"][-1] > chandelier_exit.last_chandelier_exit_short \
                and candles["close"][-2] < chandelier_exit.prev_chandelier_exit_short:
            self.signal = self.signal + 1 if self.signal > -1 else 1
        elif candles["close"][-1] < chandelier_exit.last_chandelier_exit_long \
                and candles["close"][-2] > chandelier_exit.prev_chandelier_exit_long:
            self.signal = self.signal - 1 if self.signal < 1 else -1
        else:
            self.signal = self.signal + 1 if self.signal > 0 else self.signal - 1 if self.signal != 0 else 0
        return self.signal


class ClientServices:
    def __init__(self, client):
        self.client: BinanceClient = client

    def get_client_info(self):
        class ClientInfo:
            def __init__(self, account, balances: dict):
                self.account = account
                self.balances = balances
        get_account = self.client.get_account()
        return ClientInfo(get_account, {i['asset']: float(i['free']) for i in [i for i in get_account['balances']]})

    def get_symbol_info(self, symbol: str):
        class SymbolInfoHandler:
            def __init__(self, symbol_data: dict):
                self.status = symbol_data['status']
                self.base_asset = symbol_data['baseAsset']
                self.quote_asset = symbol_data['quoteAsset']
                self.min_notional = float([i['minNotional'] for i in symbol_data['filters']
                                           if i['filterType'] == 'MIN_NOTIONAL'][0])
        return SymbolInfoHandler(self.client.get_symbol_info(symbol))

    def get_historical_candles(self, symbol: str, candle_interval, candle_type: str, from_days: int):
        if candle_type == CandleType.ClassicCandles:
            return [CandleHandler(candle).candle for candle in self.client.get_historical_klines(
                symbol=symbol, interval=candle_interval, start_str=f"{from_days} day ago UTC")]
        elif candle_type == CandleType.HeikinAshiCandles:
            historical_candles = []
            for candle in self.client.get_historical_klines(
                    symbol=symbol, interval=candle_interval, start_str=f"{from_days} day ago UTC"
            ):
                historical_candles.append(CandleHandler(candle).ha_candle(
                    historical_candles[-1] if len(historical_candles) > 0 else None))
            return historical_candles
        else:
            raise MyCustomError("Candle type not found")

    def create_stream(self, channels: list[str], markets: list[str]):
        class StreamHandler:
            def __init__(self, stream_data):
                try:
                    self.last_price = float(stream_data["price"]) if stream_data["price"] else None
                except KeyError:
                    self.last_price = None
                try:
                    self.candle = CandleHandler(stream_data["kline"]) \
                        if stream_data["kline"] and stream_data["kline"]["is_closed"] else None
                    self.live_candle = CandleHandler(stream_data["kline"]) \
                        if stream_data["kline"] and not stream_data["kline"]["is_closed"] else None
                except KeyError:
                    self.candle = None
                    self.live_candle = None

            def __str__(self):
                string = {i: str(self.__dict__[i]) if self.__dict__[i] else None for i in self.__dict__}
                return f"{string}"

        dict_channels = {"last_price": "trade", "candle_1m": "kline_1m", "candle_5m": "kline_5m",
                         "candle_15m": "kline_15m", "candle_1h": "kline_1h", "candle_1d": "kline_1d"}
        channels = [dict_channels[channel] for channel in channels]
        self.client: BinanceClient
        market_data_stream: BinanceWebSocketApiManager = BinanceWebSocketApiManager(output_default="UnicornFy")
        for channel in channels:
            market_data_stream.create_stream(channels=channel, markets=markets, output="UnicornFy")

        def market_data_stream_iterator(market_data_stream_: BinanceWebSocketApiManager):
            while True:
                oldest_stream_data_from_stream_buffer = market_data_stream_.pop_stream_data_from_stream_buffer()
                if oldest_stream_data_from_stream_buffer:
                    yield StreamHandler(oldest_stream_data_from_stream_buffer)
        return market_data_stream_iterator(market_data_stream)

    def market_order(self, symbol: str, side: str, quote_quantity=None, base_quantity=None):
        class OrderHandler:
            def __init__(self, order_data):
                self.symbol = order_data['symbol']
                self.transaction_time = Tools.milliseconds_to_datatime(int(order_data['transactTime']))
                self.quote_asset_quantity = float(order_data['cummulativeQuoteQty'])
                self.base_asset_quantity = float(order_data['executedQty'])
                self.transaction_type = order_data['type']
                self.side = order_data['side']
                self.transaction_price = float(order_data['fills'][0]['price'])
                self.commission = float(order_data['fills'][0]['commission'])
                self.status = order_data['status']

            def __str__(self):
                return str(self.__dict__)
        if quote_quantity and not base_quantity:
            return OrderHandler(self.client.order_market(symbol=symbol, side=side, quoteOrderQty=quote_quantity))
        elif base_quantity and not quote_quantity:
            return OrderHandler(self.client.order_market(symbol=symbol, side=side, quantity=base_quantity))
        else:
            raise MyCustomError(f"order_market params {quote_quantity, base_quantity} Error")


def trading_mode():
    start_datetime = datetime.now()
    print(f"session_id: {S.SESSION_ID}\nsettings: {S.dict__}")
    signal, cycle_number, order_response, live_candle, orders = 0, 0, None, None, []
    transaction_price, transaction_time, order_quote_asset_quantity = 0, None, 0
    take_profit_order_response, stop_loss_order_response = None, None
    take_profit_numb, stop_loss_numb = 0, 0
    binance_client_services = ClientServices(BinanceClient(keys.BINANCE_API_KEY, keys.BINANCE_SECRET_KEY))
    client_info = binance_client_services.get_client_info()
    symbol_info = binance_client_services.get_symbol_info(S.SYMBOL)
    quote_asset_balance, base_asset_balance = \
        client_info.balances[symbol_info.quote_asset] * S.PERCENTAGE_OF_QUOTE_ASSET_BALANCE, 0
    quote_quantity = S.POST_ORDER_QUOTE_ASSET_QUANTITY \
        if S.POST_ORDER_QUOTE_ASSET_QUANTITY <= quote_asset_balance else quote_asset_balance
    candles = CandleServices(binance_client_services.get_historical_candles(
        S.SYMBOL, S.CANDLE_INTERVAL, S.CANDLE_TYPE, S.NUMBER_OF_DAYS_AGO)).candles_dict()
    indicators = Indicators(candles)
    last_price = float(binance_client_services.client.get_ticker(symbol=S.SYMBOL)['lastPrice'])
    market_data_stream = binance_client_services.create_stream(
            channels=[S.SUBSCRIPTION_CANDLE_INTERVAL, StreamChannels.LastPrice], markets=[S.SYMBOL])
    for market_data in market_data_stream:
        if market_data.last_price:
            last_price = market_data.last_price
        if market_data.live_candle:
            if S.SUBSCRIPTION_CANDLE_TYPE == CandleType.HeikinAshiCandles:
                live_candle = market_data.live_candle.ha_candle(CandleHandler.CandleArguments(
                        candles["open"][-1], candles["high"][-1], candles["low"][-1], candles["close"][-1],
                        candles["volume"][-1], candles["datetime"][-1]))
            else:
                live_candle = market_data.live_candle.candle
        if market_data.candle:
            if S.SUBSCRIPTION_CANDLE_TYPE == CandleType.HeikinAshiCandles:
                candle = market_data.candle.ha_candle(CandleHandler.CandleArguments(
                    candles["open"][-1], candles["high"][-1], candles["low"][-1], candles["close"][-1],
                    candles["volume"][-1], candles["datetime"][-1]))
            else:
                candle = market_data.candle.candle
            for i in candles:
                candles[i] += (candle.__dict__[i],)
            indicators = Indicators(CandleServices.candles_dict_slice(candles, S.START_CANDLES_DICT_SLICE))
            indicators.chandelier_exit_calc(S.CHANDELIER_EXIT_LENGTH, S.CHANDELIER_EXIT_FACTOR)
            signal = Strategies(signal, indicators).chandelier_exit_strategy()
            if signal == 1:
                if S.REAL_ORDER:
                    order_response = binance_client_services.market_order(
                        S.SYMBOL, Sides.BUY, quote_quantity=quote_quantity)
                    base_asset_balance += order_response.base_asset_quantity
                    quote_asset_balance -= order_response.quote_asset_quantity
                    transaction_price = order_response.transaction_price
                    order_quote_asset_quantity = order_response.quote_asset_quantity
                    transaction_time = order_response.transaction_time
            elif signal == -1 and base_asset_balance > 0:
                if S.REAL_ORDER:
                    order_response = binance_client_services.market_order(
                        S.SYMBOL, Sides.SELL, base_quantity=round(base_asset_balance, 5))
                    base_asset_balance -= order_response.base_asset_quantity
                    quote_asset_balance += order_response.quote_asset_quantity
            Tools.loger(f"session_id={S.SESSION_ID}/symbol={S.SYMBOL}/order_response=False/"
                        f"take_profit_order_response=False/stop_loss_order_response=False/candle=close:"
                        f"{candles['close'][-1]},open:{candles['open'][-1]},high:{candles['high'][-1]},"
                        f"low:{candles['low'][-1]}/last_price={last_price}/signal={signal}/cycle_number="
                        f"{cycle_number}/real_order={S.REAL_ORDER}/start_datetime={start_datetime}/"
                        f"working_sec={round(abs(start_datetime - datetime.now()).total_seconds())}")
        if base_asset_balance > 0:
            if S.TAKE_PROFIT and last_price >= transaction_price + \
                    transaction_price * S.TAKE_PROFIT_NUMBERS[take_profit_numb]\
                    and order_quote_asset_quantity * S.TAKE_PROFIT_POST_ORDER_PERCENT \
                    > symbol_info.min_notional + 1 and order_quote_asset_quantity * \
                    (1 - S.TAKE_PROFIT_POST_ORDER_PERCENT) > symbol_info.min_notional + 1:
                if S.REAL_ORDER:
                    take_profit_order_response = binance_client_services.market_order(
                        S.SYMBOL, Sides.SELL, base_quantity=round(
                            base_asset_balance * S.TAKE_PROFIT_POST_ORDER_PERCENT, 5))
                    base_asset_balance -= take_profit_order_response.base_asset_quantity
                    quote_asset_balance += take_profit_order_response.quote_asset_quantity
                    order_quote_asset_quantity -= take_profit_order_response.quote_asset_quantity
                take_profit_numb += 1
            if S.STOP_LOSS and last_price < transaction_price + \
                    transaction_price * S.STOP_LOSS_NUMBERS[stop_loss_numb] \
                    and order_quote_asset_quantity * S.STOP_LOSS_POST_ORDER_PERCENT \
                    > symbol_info.min_notional + 1 and order_quote_asset_quantity * \
                    (1 - S.STOP_LOSS_POST_ORDER_PERCENT) > symbol_info.min_notional + 1 \
                    and round(abs(transaction_time - datetime.now()).total_seconds()) >= S.STOP_LOSS_TIMEOUT:
                if S.REAL_ORDER:
                    stop_loss_order_response = binance_client_services.market_order(
                        S.SYMBOL, Sides.SELL, base_quantity=round(
                            base_asset_balance * S.STOP_LOSS_POST_ORDER_PERCENT, 5))
                    base_asset_balance -= stop_loss_order_response.base_asset_quantity
                    quote_asset_balance += stop_loss_order_response.quote_asset_quantity
                    order_quote_asset_quantity -= stop_loss_order_response.quote_asset_quantity
                stop_loss_numb += 1
            if live_candle.close + (live_candle.close / 2100) < indicators.chandelier_exit.chandelier_exit_long[-1]:
                if S.REAL_ORDER:
                    order_response = binance_client_services.market_order(
                        S.SYMBOL, Sides.SELL, base_quantity=round(base_asset_balance, 5))
                    base_asset_balance -= order_response.base_asset_quantity
                    quote_asset_balance += order_response.quote_asset_quantity
        else:
            take_profit_numb, stop_loss_numb = 0, 0
        if order_response or take_profit_order_response or stop_loss_order_response:
            if order_response:
                orders = orders[S.MARTINGALE_PERIOD:] if len(orders) > S.MARTINGALE_PERIOD * 4 else orders
                orders.append(order_response)
                if len(orders) >= S.MARTINGALE_PERIOD * 2 and orders[-1].side == Sides.SELL and \
                        all([orders[i + 1].transaction_price - orders[i].transaction_price < 0
                             for i in range(S.MARTINGALE_PERIOD * -2, -1) if i % 2 == 0]):
                    quote_quantity = quote_quantity if S.MARTINGALE_PROGRESSION else S.POST_ORDER_QUOTE_ASSET_QUANTITY
                    quote_quantity += quote_quantity * S.MARTINGALE_FACTOR
                    quote_quantity = quote_quantity if quote_quantity <= quote_asset_balance else quote_asset_balance
                else:
                    quote_quantity = S.POST_ORDER_QUOTE_ASSET_QUANTITY
            order_ = order_response if order_response else take_profit_order_response \
                if take_profit_order_response else stop_loss_order_response
            Tools.loger(f"session_id={S.SESSION_ID}/symbol={order_.symbol}/"
                        f"order_response={bool(order_response)}/take_profit_order_response="
                        f"{bool(take_profit_order_response)}/stop_loss_order_response={bool(stop_loss_order_response)}/"
                        f"side={order_.side}/transaction_price={order_.transaction_price}/"
                        f"quote_asset_balance={quote_asset_balance}/base_asset_quantity="
                        f"{order_.base_asset_quantity}/base_asset_balance={base_asset_balance}/"
                        f"quote_asset_quantity={order_.quote_asset_quantity}/signal={signal}/commission="
                        f"{order_.commission}/status={order_.status}/"
                        f"transaction_time={order_.transaction_time}")
            print(f"order_response: side={order_.side}, transaction_price={order_.transaction_price}, "
                  f"base_asset_quantity={order_.base_asset_quantity}, "
                  f"quote_asset_quantity={order_.quote_asset_quantity}, "
                  f"status={order_.status}, transaction_time={order_.transaction_time}")
            order_response, take_profit_order_response, stop_loss_order_response = None, None, None
        cycle_number += 1


if __name__ == "__main__":
    trading_mode()

