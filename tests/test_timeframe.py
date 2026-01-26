from algotrading.lib.types.market_data import Timeframe, TimeframeUnit


def test_timeframe_str_and_seconds():
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    assert str(tf) == "1m"
    assert tf.seconds == 60

    tf = Timeframe(2, TimeframeUnit.HOUR)
    assert str(tf) == "2h"
    assert tf.seconds == 2 * 3600

    tf = Timeframe(1, TimeframeUnit.DAY)
    assert str(tf) == "1d"
    assert tf.seconds == 86400
