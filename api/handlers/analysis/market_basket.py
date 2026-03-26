"""handle_market_basket handler."""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_market_basket(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Simple association rule mining (support, confidence, lift) from categorical columns."""
    col = params.get("column")
    group_col = params.get("group") or params.get("transaction")
    min_support = float(params.get("min_support", 0.01))
    cats = BaseHandler.get_categorical_cols(df)
    if not col and not group_col and len(cats) >= 2:
        group_col = cats[0]
        col = cats[1]
    if not col or not group_col:
        return HandlerResult(success=False, error="Need 'group' (transaction) and 'column' (item) columns")
    baskets = df.groupby(group_col)[col].apply(set).tolist()
    n_baskets = len(baskets)
    item_counts = Counter()
    pair_counts = Counter()
    for basket in baskets:
        items = list(basket)
        for item in items:
            item_counts[item] += 1
        for pair in combinations(sorted(items), 2):
            pair_counts[pair] += 1
    rows = []
    for (a, b), count in pair_counts.most_common(50):
        support = count / n_baskets
        if support < min_support:
            continue
        conf_a_b = count / item_counts[a] if item_counts[a] > 0 else 0
        conf_b_a = count / item_counts[b] if item_counts[b] > 0 else 0
        exp = (item_counts[a] / n_baskets) * (item_counts[b] / n_baskets)
        lift = support / exp if exp > 0 else 0
        rows.append({"item_a": a, "item_b": b, "support": round(support, 4),
                      "confidence_a→b": round(conf_a_b, 4),
                      "confidence_b→a": round(conf_b_a, 4), "lift": round(lift, 2)})
    rows.sort(key=lambda r: r["lift"], reverse=True)
    return HandlerResult(success=True, result_df=pd.DataFrame(rows), output_type="query",
                         summary=f"Association rules: {len(rows)} pairs found from {n_baskets} transactions")
