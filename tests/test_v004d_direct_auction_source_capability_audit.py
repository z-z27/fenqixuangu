from __future__ import annotations

import io

import pandas as pd

from src.v004d_direct_auction_source_capability_audit import (
    EM_PRE_000001_ROWS,
    FIELD_IDS,
    OUTPUT_FILENAMES,
    SINA_QUOTE_SAMPLE,
    build_batch_stress,
    build_field_capability,
    build_historical_depth,
    build_outputs,
    parse_eastmoney_trend_row,
    parse_sina_quote_lines,
)


def test_eastmoney_premarket_extract_has_consistent_opening_match_arithmetic():
    rows = [parse_eastmoney_trend_row(row) for row in EM_PRE_000001_ROWS]
    assert [row.timestamp[-5:] for row in rows] == ["09:24", "09:25", "09:26"]
    assert rows[1].close == rows[2].close == 11.86
    assert rows[1].volume_hands == 0
    assert rows[2].volume_hands == 5094
    assert rows[2].amount_cny == rows[2].volume_hands * 100 * rows[2].close


def test_sina_quote_parser_supports_multiple_symbols_and_l1_fields():
    frame = parse_sina_quote_lines(SINA_QUOTE_SAMPLE)
    assert frame["symbol"].tolist() == ["sz000001", "sh600000"]
    assert frame.loc[0, "open"] == 11.86
    assert frame.loc[0, "bid1_price"] == 11.89
    assert frame.loc[0, "ask1_price"] == 11.90
    assert frame.loc[0, "quote_date"] == "2026-09-04"


def test_fixed_field_matrix_is_complete_for_three_source_families():
    frame = build_field_capability()
    assert len(frame) == 3 * len(FIELD_IDS)
    assert set(frame["source"]) == {"SINA", "STOCK_EASTMONEY", "AKSHARE"}
    assert frame.groupby("source")["field_id"].nunique().eq(14).all()


def test_historical_depth_tests_all_preregistered_horizons():
    frame = build_historical_depth()
    assert frame["horizon"].nunique() == 7
    assert set(frame.loc[frame["auction_data_returned"].eq("YES"), "returned_date"]) == {"2026-09-04"}
    may = frame[frame["horizon"].eq("MAY_2026")]
    assert may["auction_data_returned"].eq("NO").all()


def test_batch_stress_uses_only_fixed_sizes_and_is_not_live_latency():
    frame = build_batch_stress()
    assert set(frame["candidate_count"]) == {5, 10, 20, 30, 50}
    assert frame["test_mode"].str.contains("LIVE", case=False).all()
    sina = frame[frame["source"].eq("SINA")]
    assert sina["returned_count"].astype(int).tolist() == [5, 10, 20, 30, 50]


def test_outputs_are_complete_parseable_and_fail_closed():
    outputs, context = build_outputs()
    assert tuple(outputs) == OUTPUT_FILENAMES
    assert len(outputs) == 12
    for name in OUTPUT_FILENAMES[:-1]:
        frame = pd.read_csv(io.BytesIO(outputs[name]))
        assert len(frame) > 0, name
        assert not frame.columns.duplicated().any(), name
    review = outputs[OUTPUT_FILENAMES[-1]].decode("utf-8")
    assert context["overall_state"] == "DIRECT_AUCTION_SOURCE_PARTIAL"
    assert context["primary_source"] == "NONE"
    assert context["outcome_accessed"] == "NO"
    assert context["model_trained"] == "NO"
    assert context["buy_pass_created"] == "NO"
    assert "CURRENT_CACHE_LIMITATION_ONLY = NO" in review
    assert "OUTCOME_ACCESSED = NO" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review
