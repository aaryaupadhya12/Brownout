PRICES = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14,0.28)
}

def add_cost(traj):
    t = traj.totals
    t["api_cost_usd"] = cost_usd(traj.model_id, t.get("input_tokens", 0),
                                 t.get("output_tokens", 0), t.get("reasoning_tokens", 0))
    t["price_per_mtok"] = price_of(traj.model_id)

def price_of(model_id):
    return PRICES.get(model_id)

def cost_usd(model_id, input_tokens, output_tokens, reasoning_tokens=0):
    p = PRICES.get(model_id)
    if p is None:
        return None
    return (input_tokens * p[0] + (output_tokens + reasoning_tokens) * p[1]) / 1_000_000

