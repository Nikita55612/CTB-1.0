import random


class CandleType:
    ClassicCandles: str = "cl"
    HeikinAshiCandles: str = "ha"


class SubscriptionCandleInterval:
    Candle_1M = "candle_1m"
    Candle_5M = "candle_5m"
    Candle_15M = "candle_15m"
    CandleHour = "candle_1h"
    CandleDay = "candle_1d"


SYMBOL: str = "BTCUSDT"
REAL_ORDER: bool = False
NUMBER_OF_DAYS_AGO: int = 1
POST_ORDER_QUOTE_ASSET_QUANTITY: int = 120
PERCENTAGE_OF_QUOTE_ASSET_BALANCE: float = 0.64
CANDLE_TYPE: str = CandleType.HeikinAshiCandles
SUBSCRIPTION_CANDLE_TYPE: str = CandleType.HeikinAshiCandles
CANDLE_INTERVAL: str = "1m"
SUBSCRIPTION_CANDLE_INTERVAL = SubscriptionCandleInterval.Candle_1M
MARTINGALE_PERIOD: int = 3
MARTINGALE_FACTOR: float = 0.2
MARTINGALE_PROGRESSION: bool = True
CHANDELIER_EXIT_LENGTH: int = 2
CHANDELIER_EXIT_FACTOR: float = 1.18
START_CANDLES_DICT_SLICE: int = {2: -128, 3: -164, 4: -256, 5: 312}[CHANDELIER_EXIT_LENGTH] \
    if CHANDELIER_EXIT_LENGTH < 6 else 0
STOP_LOSS: bool = True
STOP_LOSS_NUMBERS: tuple = (-0.0002, -1)
STOP_LOSS_POST_ORDER_PERCENT: float = 0.22
STOP_LOSS_TIMEOUT: int = 140  # sec
TAKE_PROFIT: bool = True
TAKE_PROFIT_NUMBERS: tuple = (0.0017, 0.0034, 0.0075, 1)
TAKE_PROFIT_POST_ORDER_PERCENT: float = 0.12
dir__ = {i for i in dir() if not i.startswith("__") and i not in ('random', 'SubscriptionCandleInterval', 'CandleType')}
dict__ = {i: globals()[i] for i in dir__}
SESSION_ID = [i for i in
              "PyVBUkDEKMxcwGIoJOuiTXgenlFrSYWdqsZzjHRhtvfmCLbNapAQ"[random.randrange(0, 51)] +
              "SmauEfRHijrtZnoqwsVOYxIQAMlTgydbCWXeNzhUKFPpDGkJBcvL"[random.randrange(0, 51)] +
              "lvnobfOiPBFMZJHyLpWRkGEQsYNwgDdUtuqIejCzKrXATVSxahmc"[random.randrange(0, 51)] +
              "uXwkfdijHJhaZAxzCUFEieNOynLSQtlPqBGRTKVvIosWYMDpcgbr"[random.randrange(0, 51)] +
              "GWfbgTuRCocvlwmUiesMQLhXYIKElAVkOJFHdyxzDPqantZrjBNp"[random.randrange(0, 51)] +
              "LzlSHVywOEeIpBQmoRXWFNkPfvhZKgjGTsaUJxMcrbiqnutYDCAd"[random.randrange(0, 51)] +
              "rWiklbmBQHROVdqtJxuMwcahzGsSpefDNYZjEgUKXTIAovLPFCny"[random.randrange(0, 51)] +
              "MtIlFwnvEcjQmPrdSgJxhyKZsbGoLfVzRAWkTCuOiUYXeDaBNHpq"[random.randrange(0, 51)] +
              str(random.randrange(1000000000, 9999999999))]
for i in range(random.randrange(2, 10)):
    random.shuffle(SESSION_ID)
SESSION_ID = "".join(SESSION_ID)
