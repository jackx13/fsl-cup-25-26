#!/usr/bin/env python3
import random
from typing import List, Dict
import json
import os
import requests
import streamlit as st
import pandas as pd


FPL_BASE = "https://fantasy.premierleague.com/api"

# Default values for your specific league / people
DEFAULT_LEAGUE_ID = 2448
DEFAULT_RADOSTIN_ENTRY_ID = 101913  # Kai me a reiver
DEFAULT_KRISTIYAN_ENTRY_ID = 5642535
CUP_GROUP_START_GW = 22
CUP_GROUP_ROUNDS = 6
CUP_ROUND_TO_GW = {rnd: CUP_GROUP_START_GW + (rnd - 1) for rnd in range(1, CUP_GROUP_ROUNDS + 1)}

# ---------- FPL helpers ----------

DRAW_FILE = os.path.join(os.path.dirname(__file__), "cup_draw_league2448_gw22_groups.json")

def save_draw_to_file(groups: dict, fixtures: dict | None = None):
    payload = {
        "groups": groups,
        "fixtures": fixtures,
    }
    with open(DRAW_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_draw_from_file():
    if not os.path.exists(DRAW_FILE):
        return None, None
    with open(DRAW_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("groups"), payload.get("fixtures")

@st.cache_data(ttl=60)
def get_event_live_element_points(gw: int) -> dict[int, int]:
    """Map element_id -> live total_points for a given GW."""
    live_url = f"{FPL_BASE}/event/{gw}/live/"
    r = requests.get(live_url, timeout=30)
    r.raise_for_status()
    data = r.json()

    elem_points = {}
    for el in data.get("elements", []):
        elem_points[el["id"]] = int(el.get("stats", {}).get("total_points", 0))
    return elem_points


@st.cache_data(ttl=60)
def get_entry_true_live_net_score(entry_id: int, gw: int) -> int | None:
    """
    True live net score:
      net = sum(live_player_points * pick_multiplier) - transfer_hits
    """
    picks_url = f"{FPL_BASE}/entry/{entry_id}/event/{gw}/picks/"
    pr = requests.get(picks_url, timeout=20)
    if pr.status_code != 200:
        return None
    pdata = pr.json()

    entry_hist = pdata.get("entry_history")
    picks = pdata.get("picks", [])
    if not entry_hist or not picks:
        return None

    transfer_cost = int(entry_hist.get("event_transfers_cost", 0))
    elem_points = get_event_live_element_points(gw)

    gross = 0
    for p in picks:
        element = int(p["element"])
        mult = int(p.get("multiplier", 0))  # captain/TC etc included
        gross += elem_points.get(element, 0) * mult

    return gross - transfer_cost

import time

@st.cache_data(ttl=60)
def get_current_gw() -> int | None:
    """Returns current GW from bootstrap-static (live during the GW)."""
    data = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=20).json()
    for ev in data.get("events", []):
        if ev.get("is_current"):
            return ev.get("id")
    # fallback: latest finished
    finished = [e["id"] for e in data.get("events", []) if e.get("finished")]
    return max(finished) if finished else None


@st.cache_data(ttl=60)
def get_entry_live_net_score(entry_id: int, gw: int) -> int | None:
    """
    Live net score for cup:
      net = entry_history.points - entry_history.event_transfers_cost
    Uses /entry/{id}/event/{gw}/picks/ which updates live during the GW.
    """
    url = f"{FPL_BASE}/entry/{entry_id}/event/{gw}/picks/"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    eh = data.get("entry_history")
    if not eh:
        return None
    return int(eh.get("points", 0)) - int(eh.get("event_transfers_cost", 0))

def get_league_standings(league_id: int) -> List[Dict]:
    """
    Fetch classic league standings from FPL API and return a list of players
    sorted by rank ascending.
    Only the 'results' array is used.
    """
    all_results = []
    page = 1
    while True:
        url = f"{FPL_BASE}/leagues-classic/{league_id}/standings/"
        resp = requests.get(url, params={"page_standings": page}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        standings = data.get("standings", {})
        results = standings.get("results", [])
        all_results.extend(results)

        if not standings.get("has_next"):
            break
        page += 1

    # Normalize shape: we only really need rank, entry, player_name, entry_name
    cleaned = []
    for r in all_results:
        cleaned.append(
            {
                "rank": r.get("rank"),
                "entry_id": r.get("entry"),
                "manager_name": r.get("player_name"),
                "team_name": r.get("entry_name"),
            }
        )

    # Sort by rank just in case
    cleaned.sort(key=lambda x: x["rank"])
    return cleaned


# ---------- Cup logic (pots + draw) ----------

def clone_player(p: Dict) -> Dict:
    """Return a shallow copy so we can duplicate Kristiyan."""
    return {
        "rank": p["rank"],
        "entry_id": p["entry_id"],
        "manager_name": p["manager_name"],
        "team_name": p["team_name"],
    }


def build_pots(
    standings: List[Dict],
    radostin_id: int,
    kristiyan_id: int,
) -> (List[Dict], List[Dict], List[Dict], List[Dict]):

    if len(standings) < 32:
        raise ValueError("Need at least 32 teams in the league for the Cup.")

    # Use list positions instead of rank numbers (more robust)
    top32 = standings[:32]  # already sorted by rank by get_league_standings()

    # Helper to clone player dict so Kristiyan can appear twice
    def clone_player(p: Dict) -> Dict:
        return {
            "rank": p.get("rank"),
            "entry_id": p["entry_id"],
            "manager_name": p["manager_name"],
            "team_name": p["team_name"],
        }
    pot1 = pot2 = pot3 = pot4 = None

    if not st.session_state.get("draw_locked", False):
        try:
            standings = get_league_standings(int(league_id))
            pot1, pot2, pot3, pot4 = build_pots(
                standings=standings,
                radostin_id=int(radostin_entry_id),
                kristiyan_id=int(kristiyan_entry_id),
            )
        except Exception as e:
            st.error(f"Error building pots: {e}")
            st.stop()  
    # Positions are 0-based:
    # placed 1–8   => [0:8]
    # placed 9–16  => [8:16]
    # placed 17–24 => [16:24]
    # placed 25th  => index 24
    # placed 26–32 => [25:32]
    pot1 = top32[0:8]
    pot2 = top32[8:16]

    pot3_candidates = top32[16:24]
    p25 = top32[24]

    pot3 = [p for p in pot3_candidates if p["entry_id"] != radostin_id]
    if len(pot3) < 8:
        # We excluded Radostin -> add 25th placed to fill
        if p25["entry_id"] == radostin_id:
            raise ValueError("Radostin is in 25th place too; adjust rules.")
        pot3.append(p25)

    if len(pot3) != 8:
        raise ValueError(f"Pot 3 has {len(pot3)} entries (expected 8).")

    pot4 = top32[25:32]  # 26th–32nd placed

    # Ensure Radostin isn't in pot4 (just in case)
    pot4 = [p for p in pot4 if p["entry_id"] != radostin_id]

    # Find Kristiyan anywhere in league standings
    kristiyan = next((p for p in standings if p["entry_id"] == kristiyan_id), None)
    if kristiyan is None:
        raise ValueError("Kristiyan not found in the league standings. Check the entry ID.")

    pot4.append(clone_player(kristiyan))  # duplicate entry for Kristiyan

    if len(pot4) != 8:
        raise ValueError(f"Pot 4 has {len(pot4)} entries (expected 8).")

    # Final safety: Radostin must not appear
    for pot in (pot1, pot2, pot3, pot4):
        for p in pot:
            if p["entry_id"] == radostin_id:
                raise ValueError("Radostin is still present in a pot.")

    return pot1, pot2, pot3, pot4


def draw_groups(pot1, pot2, pot3, pot4, seed: int | None = None) -> Dict[str, Dict[str, Dict]]:
    """
    Input: 4 pots of 8 players each.
    Output:
      {
        "A": {"pot1": {...}, "pot2": {...}, "pot3": {...}, "pot4": {...}},
        ...
        "H": {...}
      }
    """

    if seed is not None:
        random.seed(seed)

    # Work on copies
    pots = [pot1[:], pot2[:], pot3[:], pot4[:]]
    for pot in pots:
        random.shuffle(pot)

    groups = {}
    group_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i, g in enumerate(group_names):
        groups[g] = {
            "pot1": pots[0][i],
            "pot2": pots[1][i],
            "pot3": pots[2][i],
            "pot4": pots[3][i],
        }
    return groups


# ---------- Streamlit UI ----------
import time
import pandas as pd

@st.cache_data(ttl=3600)
def get_entry_history(entry_id: int) -> dict:
    """Fetch entry history once and cache it."""
    url = f"{FPL_BASE}/entry/{entry_id}/history/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def get_net_from_history(history_json: dict, gw: int) -> int | None:
    """Compute net = points - transfer_cost for a specific GW from cached history JSON."""
    for row in history_json.get("current", []):
        if int(row.get("event")) == int(gw):
            pts = int(row.get("points", 0))
            cost = int(row.get("event_transfers_cost", 0))
            return pts - cost
    return None
    
def precompute_net_scores(groups: dict, gws: list[int]) -> dict[tuple[int, int], int | None]:
    """
    Returns {(entry_id, gw): net_score_or_None}
    Fetches each entry history once (cached).
    """
    entry_ids = set()
    for g in groups.values():
        for pot_key in ("pot1", "pot2", "pot3", "pot4"):
            entry_ids.add(int(g[pot_key]["entry_id"]))

    cache = {}
    for entry_id in entry_ids:
        hist = get_entry_history(entry_id)  # cached
        for gw in gws:
            cache[(entry_id, gw)] = get_net_from_history(hist, gw)

    return cache


def animate_draw(pot1, pot2, pot3, pot4, seed: int | None = None):
    """
    Animated version of draw:
    - Shuffle each pot
    - Reveal group by group (A->H), pot by pot
    """
    if seed is not None:
        random.seed(seed)

    pots = [pot1[:], pot2[:], pot3[:], pot4[:]]
    for pot in pots:
        random.shuffle(pot)

    group_names = list("ABCDEFGH")
    groups = {g: {} for g in group_names}

    slot = st.empty()
    for i, g in enumerate(group_names):
        # reveal Pot 1..4 with short pauses
        groups[g]["pot1"] = pots[0][i]
        slot.info(f"Group {g}: Pot 1 → {pots[0][i]['manager_name']} ({pots[0][i]['team_name']})")
        time.sleep(0.45)

        groups[g]["pot2"] = pots[1][i]
        slot.info(f"Group {g}: Pot 2 → {pots[1][i]['manager_name']} ({pots[1][i]['team_name']})")
        time.sleep(0.45)

        groups[g]["pot3"] = pots[2][i]
        slot.info(f"Group {g}: Pot 3 → {pots[2][i]['manager_name']} ({pots[2][i]['team_name']})")
        time.sleep(0.45)

        groups[g]["pot4"] = pots[3][i]
        slot.success(f"Group {g} complete ✅")
        time.sleep(0.45)

    slot.empty()
    return groups


def build_group_teams(groups_for_one_group: dict) -> list[dict]:
    """Return list of 4 team dicts in pot order."""
    return [groups_for_one_group["pot1"], groups_for_one_group["pot2"], groups_for_one_group["pot3"], groups_for_one_group["pot4"]]


def round_robin_4_double(teams: list[dict]):
    """
    4 teams, double round-robin => 6 rounds, 2 matches per round.
    Uses a standard circle method for the first 3 rounds, then reverse fixtures.
    Returns: list of rounds, each round = list of matches (home, away)
    """
    assert len(teams) == 4
    t = teams[:]
    # circle method for 4:
    # rounds (single):
    # R1: 1v4, 2v3
    # R2: 1v3, 4v2
    # R3: 1v2, 3v4
    r1 = [(t[0], t[3]), (t[1], t[2])]
    r2 = [(t[0], t[2]), (t[3], t[1])]
    r3 = [(t[0], t[1]), (t[2], t[3])]
    first_leg = [r1, r2, r3]
    second_leg = [[(away, home) for (home, away) in rnd] for rnd in first_leg]
    return first_leg + second_leg


def build_all_fixtures(groups: dict):
    """
    Builds fixtures for all groups A-H.
    Output format:
      fixtures[group] = [
        {"round": 1, "home": team_dict, "away": team_dict},
        ...
      ]
    """
    fixtures = {}
    for g, members in groups.items():
        teams = build_group_teams(members)
        rounds = round_robin_4_double(teams)

        group_matches = []
        for rnd_i, rnd in enumerate(rounds, start=1):
            for (home, away) in rnd:
                group_matches.append({"round": rnd_i, "home": home, "away": away})
        fixtures[g] = group_matches
    return fixtures


def compute_group_standings(groups: dict, fixtures: dict, round_to_gw: dict[int, int], net_cache: dict):
    """
    Returns dict[group] -> DataFrame with standings.
    Also returns dict[group] -> match results DataFrame.
    """
    standings_out = {}
    results_out = {}

    for g in groups.keys():
        # init table
        teams = build_group_teams(groups[g])
        table = {}
        for team in teams:
            table[team["entry_id"]] = {
                "Entry ID": team["entry_id"],
                "Manager": team["manager_name"],
                "Team": team["team_name"],
                "P": 0, "W": 0, "D": 0, "L": 0,
                "Pts": 0,
                "NetScoreFor": 0,
                "NetScoreAgainst": 0,
                "NetDiff": 0,
            }

        match_rows = []

        for m in fixtures[g]:
            rnd = m["round"]
            gw = round_to_gw.get(rnd)
            if gw is None:
                continue

            h = m["home"]
            a = m["away"]

            h_net = net_cache.get((int(h["entry_id"]), int(gw)))
            a_net = net_cache.get((int(a["entry_id"]), int(gw)))

            # If GW not available yet, skip match in standings
            if h_net is None or a_net is None:
                match_rows.append({
                    "Round": rnd, "GW": gw,
                    "Home": f"{h['manager_name']} ({h['team_name']})",
                    "Away": f"{a['manager_name']} ({a['team_name']})",
                    "Home net": None, "Away net": None,
                    "Result": "—",
                })
                continue

            # update played + net “for/against”
            for entry_id in (h["entry_id"], a["entry_id"]):
                table[entry_id]["P"] += 1

            table[h["entry_id"]]["NetScoreFor"] += h_net
            table[h["entry_id"]]["NetScoreAgainst"] += a_net
            table[a["entry_id"]]["NetScoreFor"] += a_net
            table[a["entry_id"]]["NetScoreAgainst"] += h_net

            # decide W/D/L + points
            if h_net > a_net:
                table[h["entry_id"]]["W"] += 1
                table[a["entry_id"]]["L"] += 1
                table[h["entry_id"]]["Pts"] += 3
                res = "H"
            elif h_net < a_net:
                table[a["entry_id"]]["W"] += 1
                table[h["entry_id"]]["L"] += 1
                table[a["entry_id"]]["Pts"] += 3
                res = "A"
            else:
                table[h["entry_id"]]["D"] += 1
                table[a["entry_id"]]["D"] += 1
                table[h["entry_id"]]["Pts"] += 1
                table[a["entry_id"]]["Pts"] += 1
                res = "D"

            match_rows.append({
                "Round": rnd, "GW": gw,
                "Home": f"{h['manager_name']} ({h['team_name']})",
                "Away": f"{a['manager_name']} ({a['team_name']})",
                "Home net": h_net, "Away net": a_net,
                "Result": res,
            })

        # finalize NetDiff
        for entry_id, row in table.items():
            row["NetDiff"] = row["NetScoreFor"] - row["NetScoreAgainst"]

        df = pd.DataFrame(table.values())
        df = df.sort_values(by=["Pts", "NetDiff", "NetScoreFor"], ascending=False).reset_index(drop=True)

        standings_out[g] = df
        results_out[g] = pd.DataFrame(match_rows)

    return standings_out, results_out
    
    


def main():
    st.set_page_config(page_title="Купата — FPL Cup Draw", layout="wide")
    

    st.title("🏆 Купата — FPL Cup Draw")

        st.sidebar.header("Настройки")
    league_id = st.sidebar.number_input(
        "League ID",
        min_value=1,
        value=DEFAULT_LEAGUE_ID,
        step=1,
    )

    radostin_entry_id = st.sidebar.number_input(
        "Radostin (excluded) entry ID",
        min_value=1,
        value=DEFAULT_RADOSTIN_ENTRY_ID,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("👤 Kristiyan Kovachev (double entry)")

    kristiyan_entry_id = st.sidebar.number_input(
        "Kristiyan entry ID",
        min_value=1,
        value=DEFAULT_KRISTIYAN_ENTRY_ID,
        step=1,
        help="Enter the FPL entry ID for Kristiyan Kovachev. He will appear twice in Pot 4.",
    )

    seed_value = st.sidebar.number_input(
        "Random seed (optional, for reproducible draw)",
        value=2448,
        step=1,
    )

    # --- Load locked draw (production mode) ---
    st.session_state.setdefault("groups", None)
    st.session_state.setdefault("fixtures", None)
    st.session_state.setdefault("draw_locked", False)

    file_groups, file_fixtures = load_draw_from_file()

    if file_groups:
        st.session_state["groups"] = file_groups
        st.session_state["fixtures"] = file_fixtures or build_all_fixtures(file_groups)
        st.session_state["draw_locked"] = True

    if kristiyan_entry_id <= 0:
        st.error("Please enter a valid entry ID for Kristiyan Kovachev.")
        return

    # --- Only fetch standings in setup mode (when NOT locked) ---
    standings = None
    if not st.session_state["draw_locked"]:
        with st.spinner("Fetching league standings from FPL..."):
            try:
                standings = get_league_standings(int(league_id))
            except Exception as e:
                st.error(f"Error fetching standings: {e}")
                return

        if len(standings) < 32:
            st.error(f"League has only {len(standings)} teams; need at least 32 for the Cup.")
            return

        st.subheader("Лига — Top 32 (before Cup rules)")
        st.dataframe(
            [
                {
                    "Rank": p["rank"],
                    "Entry ID": p["entry_id"],
                    "Manager": p["manager_name"],
                    "Team": p["team_name"],
                }
                for p in standings[:32]
            ],
            use_container_width=True,
        )
    else:
        st.sidebar.success("🔒 Cup draw is locked (using saved JSON).")

    # Build pots
    try:
        pot1, pot2, pot3, pot4 = build_pots(
            standings=standings,
            radostin_id=int(radostin_entry_id),
            kristiyan_id=int(kristiyan_entry_id),
        )
    except Exception as e:
        st.error(f"Error building pots: {e}")
        return        
    tab_draw, tab_fix, tab_results, tab_table = st.tabs(["🎲 Draw", "🗓 Fixtures", "📈 Results", "📊 Standings"])
        # Initialize session state keys
    st.session_state.setdefault("groups", None)
    st.session_state.setdefault("fixtures", None)
    st.session_state.setdefault("draw_locked", False)
    st.session_state.setdefault("round_to_gw", None)    
    # Get groups from session if already drawn
    groups = st.session_state.get("groups")
    fixtures = st.session_state.get("fixtures")
    st.session_state.setdefault("groups", None)
    st.session_state.setdefault("fixtures", None)
    st.session_state.setdefault("draw_locked", False)
    # Always try to load locked draw first
    file_groups, file_fixtures = load_draw_from_file()
    if file_groups:
        st.session_state["groups"] = file_groups
        st.session_state["fixtures"] = file_fixtures or build_all_fixtures(file_groups)
        st.session_state["draw_locked"] = True
    else:
        st.session_state.setdefault("draw_locked", False)
    with tab_draw:
        st.subheader("🎲 Draw (Locked)")

        # If draw is already loaded from file, just show it
        if st.session_state.get("groups"):
            st.success("🔒 Official Cup Draw is locked and loaded from file.")

        else:
            st.warning("Draw not found. This should only happen once (setup phase).")

            if st.button("✅ Create and lock draw"):
                groups = file_groups if file_groups else draw_groups(pot1, pot2, pot3, pot4, seed=int(seed_value))
                fixtures = build_all_fixtures(groups)

                st.session_state["groups"] = groups
                st.session_state["fixtures"] = fixtures

                save_draw_to_file(groups, fixtures)

                st.success("Draw created and permanently locked ✅")
                st.rerun()

        # Always display groups if they exist
        groups = st.session_state.get("groups")
        if groups:
            pot_order = [("pot1", "Pot 1"), ("pot2", "Pot 2"), ("pot3", "Pot 3"), ("pot4", "Pot 4")]

            for g in list("ABCDEFGH"):
                st.markdown(f"### Group {g}")
                rows = []
                for i, (pot_key, pot_label) in enumerate(pot_order, start=1):
                    p = groups[g][pot_key]
                    rows.append({
                         "#": i,
                        "Pot": pot_label,
                        "Rank": p.get("rank"),
                        "Entry ID": p.get("entry_id"),
                        "Manager": p.get("manager_name"),
                        "Team": p.get("team_name"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



    with tab_fix:
        st.subheader("🗓 Fixtures")

        if not st.session_state.get("fixtures"):
            st.info("Run the draw first (Draw tab).")
        else:
            fixtures = st.session_state["fixtures"]

            round_to_gw = CUP_ROUND_TO_GW
            st.session_state["round_to_gw"] = round_to_gw
            st.caption(f"Group Stage is hardcoded: GW{CUP_GROUP_START_GW}–GW{CUP_GROUP_START_GW + CUP_GROUP_ROUNDS - 1}")
            st.caption(f"Round → GW mapping: {round_to_gw}")

            for g in list("ABCDEFGH"):
                st.markdown(f"### Group {g}")
                rows = []
                for m in fixtures[g]:
                    rows.append({
                        "Round": m["round"],
                        "GW": round_to_gw[m["round"]],
                        "Home": f"{m['home']['manager_name']} ({m['home']['team_name']})",
                        "Away": f"{m['away']['manager_name']} ({m['away']['team_name']})",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                
    with tab_results:
        st.subheader("📈 Results (Live GW)")

        # 1) Always define GW first
        gw = get_current_gw()
        st.write(f"Current GW: {gw}")

        # Refresh button
        if st.button("🔄 Refresh live data"):
            st.cache_data.clear()
            st.rerun()

        if gw is None:
            st.error("Could not determine current GW.")
            st.stop()

        # 2) Guards
        if not st.session_state.get("groups") or not st.session_state.get("fixtures"):
            st.info("Run the draw first (Draw tab).")
            st.stop()

        # 3) Cup group stage hardcoded: GW22–GW27
        CUP_START_GW = 22
        CUP_END_GW = 27

        if gw < CUP_START_GW or gw > CUP_END_GW:
            st.warning(f"Current GW ({gw}) is outside the Cup group stage (GW{CUP_START_GW}–GW{CUP_END_GW}).")
            st.stop()

        round_for_gw = gw - (CUP_START_GW - 1)  # GW22->1 ... GW27->6
        st.success(f"Showing Cup **Round {round_for_gw}** (GW {gw})")

        # Optional debug (now safe)
        st.caption(f"DEBUG Kristiyan live net: {get_entry_true_live_net_score(5642535, gw)}")

        fixtures = st.session_state["fixtures"]

        # 4) Render results per group for this round
        for g in list("ABCDEFGH"):
            st.markdown(f"### Group {g} — Round {round_for_gw}")

            matches = [m for m in fixtures[g] if int(m["round"]) == int(round_for_gw)]
            rows = []

            for m in matches:
                h = m["home"]
                a = m["away"]

                h_net = get_entry_true_live_net_score(h["entry_id"], gw)
                a_net = get_entry_true_live_net_score(a["entry_id"], gw)

                if h_net is None or a_net is None:
                    res = "—"
                elif h_net > a_net:
                    res = "Home win"
                elif h_net < a_net:
                    res = "Away win"
                else:
                    res = "Draw"

                rows.append({
                    "Home": f"{h['manager_name']} ({h['team_name']})",
                    "Away": f"{a['manager_name']} ({a['team_name']})",
                    "Home net (GW pts - hits)": h_net,
                    "Away net (GW pts - hits)": a_net,
                    "Result": res,
                    })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



    with tab_table:
        st.subheader("📊 Standings")

        if not st.session_state.get("groups") or not st.session_state.get("fixtures"):
            st.info("Run the draw first (Draw tab).")
            st.stop()
            
        if st.button("🔄 Refresh standings data"):
            st.cache_data.clear()
            st.rerun()

        # ✅ Hardcoded group-stage mapping
        round_to_gw = {1: 22, 2: 23, 3: 24, 4: 25, 5: 26, 6: 27}
        gws = list(round_to_gw.values())

        with st.spinner("Fetching histories (cached) and computing tables..."):
            net_cache = precompute_net_scores(st.session_state["groups"], gws)

        standings_out, results_out = compute_group_standings(
            st.session_state["groups"],
            st.session_state["fixtures"],
            round_to_gw,
            net_cache,
        )

        for g in list("ABCDEFGH"):
            st.markdown(f"### Group {g} — Table")
            st.dataframe(standings_out[g], use_container_width=True, hide_index=True)

            with st.expander("Match results"):
                st.dataframe(results_out[g], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
