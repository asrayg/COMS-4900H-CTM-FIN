import numpy as np

def apply_spread_shock(bid_px, ask_px, bid_sz, ask_sz, factor=3.0, duration_steps=40):
    """
    Simulates a sudden spread widening shock.
    Expects bid_px, ask_px of shape (steps, K).
    """
    shocked_bid = bid_px.copy()
    shocked_ask = ask_px.copy()
    
    center = len(bid_px) // 2
    start = max(0, center - duration_steps // 2)
    end = min(len(bid_px), center + duration_steps // 2)
    
    for i in range(start, end):
        spread = shocked_ask[i, 0] - shocked_bid[i, 0]
        new_spread = spread * factor
        
        # Symmetrically widen the spread
        shocked_ask[i, 0] = shocked_bid[i, 0] + new_spread / 2
        shocked_bid[i, 0] = shocked_ask[i, 0] - new_spread / 2
        
    return shocked_bid, shocked_ask, bid_sz, ask_sz

def apply_liquidity_cliff(bid_px, ask_px, bid_sz, ask_sz, duration_steps=40):
    """
    Simulates a sudden drop in liquidity at the top of the book.
    """
    shocked_bid_sz = bid_sz.copy()
    shocked_ask_sz = ask_sz.copy()
    
    center = len(bid_sz) // 2
    start = max(0, center - duration_steps // 2)
    end = min(len(bid_sz), center + duration_steps // 2)
    
    # Wipe out top 3 levels of size
    for i in range(start, end):
        shocked_bid_sz[i, :3] = 0
        shocked_ask_sz[i, :3] = 0
        
    return bid_px, ask_px, shocked_bid_sz, shocked_ask_sz
