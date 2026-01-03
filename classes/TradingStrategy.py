from dataclasses import dataclass
from typing import Callable, Sequence

StrategyRule = Callable[[Sequence[float], float, float], int]


@dataclass
class TradingStrategy:
    """
    Configurable trading strategy driven by a user-supplied rule.
    """
    rule: StrategyRule

    def decide(self, price_history: Sequence[float], current_price: float, predicted_price: float) -> int:
        """
        Decide the position for the next period.

        Args:
            price_history: Sequence of historical prices (includes current_price).
            current_price: Current asset price.
            predicted_price: Model-predicted future price.

        Returns:
            -1 for short, 0 for cash, 1 for long.
        """
        return int(self.rule(price_history, current_price, predicted_price))

    @staticmethod
    def predicted_vs_close(threshold: float = 0.005) -> "TradingStrategy":
        """
        Long if predicted > current * (1 + threshold), short if below (1 - threshold).
        """
        def _rule(_history: Sequence[float], current: float, predicted: float) -> int:
            if current <= 0.0:
                return 0
            if predicted > current * (1.0 + threshold):
                return 1
            if predicted < current * (1.0 - threshold):
                return -1
            return 0

        return TradingStrategy(rule=_rule)

    @staticmethod
    def prediction_direction() -> "TradingStrategy":
        """
        Long if predicted > current, short if predicted < current.
        """
        def _rule(_history: Sequence[float], current: float, predicted: float) -> int:
            if predicted > current:
                return 1
            if predicted < current:
                return -1
            return 0

        return TradingStrategy(rule=_rule)

    @staticmethod
    def trend_with_prediction(lookback: int = 3) -> "TradingStrategy":
        """
        Long if last lookback days are up and predicted > current.
        Short if last lookback days are down and predicted < current.
        """
        def _rule(history: Sequence[float], current: float, predicted: float) -> int:
            if len(history) < lookback + 1:
                return 0
            window = history[-(lookback + 1):]
            ups = all(b > a for a, b in zip(window[:-1], window[1:]))
            downs = all(b < a for a, b in zip(window[:-1], window[1:]))
            if ups and predicted > current:
                return 1
            if downs and predicted < current:
                return -1
            return 0

        return TradingStrategy(rule=_rule)
