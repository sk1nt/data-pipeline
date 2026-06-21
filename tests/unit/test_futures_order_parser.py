from src.services.futures_order_parser import FuturesOrderParser, FuturesAction


def test_parse_flat_allows_zero_tp_ticks():
    parser = FuturesOrderParser()

    params = parser.parse("flat", "/MNQZ5", 0, 1, "dry")

    assert params.action == FuturesAction.FLAT
    assert params.symbol == "/MNQZ5"
    assert params.tp_ticks == 0
    assert params.quantity == 1
    assert params.mode == "dry"
