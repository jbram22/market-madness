import requests
import numpy as np
import pandas as pd

COLS = ["quarter_finals", "semi_finals", "finals", "championship"]


# helper to get team name from question
def extract_team(question):
    return question.split("Will ")[1].split(" advance to")[0]


# grab probabilities associated with particular round
def construct_round_map(slug):
    response = requests.get(f"https://gamma-api.polymarket.com/events/slug/{slug}")
    event = response.json()

    hashmap = {}
    for market in event.get("markets", []):
        if (not market["closed"]) and market['active']:
            question = market.get('question')
            if "Team" not in question:
                bid = float(market["bestBid"])
                ask = float(market["bestAsk"])
                mid = (bid + ask) / 2
                if "winner" in slug:
                    team = question.split("Will ")[1].split(" win")[0]
                else:
                    team = extract_team(question)
                hashmap[team] = mid
    return hashmap


# concatenate data into pandas dataframe 
# will need to update as tourny progresses, 
# but same w rest of code really
def concatenate_data(qf_map, sf_map, f_map, chip_map):
    import pandas as pd

    # Use sf_map as the base universe
    teams = sorted(sf_map.keys())

    df = pd.DataFrame({
        "team": teams,
        "quarter_finals": [qf_map.get(t, 0) if t in qf_map else 1 for t in teams],
        "semi_finals": [sf_map.get(t, 0) for t in teams],
        "finals": [f_map.get(t, 0) for t in teams],
        "championship": [chip_map.get(t, 0) for t in teams]
    })

    return df


# smooth probabilities
def smooth_with_min_gap(row):
    vals = row[COLS].values.copy()
    
    # eps = 0.02 * vals[0]  # 2% of QF probability
    eps = 0.01
    
    # Step 1: enforce monotonicity + minimum gaps
    for i in range(1, len(vals)):
        vals[i] = min(vals[i], vals[i-1] - eps)
    
    # Step 2: keep valid range
    vals = np.clip(vals, 0, 1)
    
    return vals


def compute_ev_df():
    # gather maps
    qf_map = construct_round_map("ncaa-tournament-team-to-make-quarterfinals")
    sf_map = construct_round_map("ncaa-tournament-team-to-make-semifinals")
    f_map = construct_round_map("ncaa-tournament-team-to-make-national-championship")
    chip_map = construct_round_map("2026-ncaa-tournament-winner")

    # concatenate
    df = concatenate_data(qf_map, sf_map, f_map, chip_map)

    # smooth probabilities
    df[COLS] = np.vstack(df.apply(smooth_with_min_gap, axis=1))
    assert (df["quarter_finals"] > df["semi_finals"]).all()
    assert (df["semi_finals"] > df["finals"]).all()
    assert (df["finals"] > df["championship"]).all()

    # compute p(losing in particular round)
    df["lose_in_qf"] = df["quarter_finals"] - df["semi_finals"]
    df["lose_in_sf"] = df["semi_finals"] - df["finals"]
    df["lose_in_finals"] = df["finals"] - df["championship"]


    df["total_prob"] = (
        (1 - df["quarter_finals"]) +   # lose before QF
        df["lose_in_qf"] +
        df["lose_in_sf"] +
        df["lose_in_finals"] +
        df["championship"]
    )

    # ensure probs sum to 1
    assert (df["total_prob"] - 1 < 1e-13).all()

    df["EV"] = (
        4 * (1 - df["quarter_finals"]) +
        8 * df["lose_in_qf"] +
        16 * df["lose_in_sf"] +
        32 * df["lose_in_finals"] +
        64 * df["championship"]
    )

    return df









