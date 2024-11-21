import json
import numpy as np
from difflib import get_close_matches

# 讀取球員和戰績資料
with open('P_Players_performance_23_24.json', 'r', encoding='utf-8') as f:
    players_data_p = json.load(f)

with open('T1_Players_performance_23_24.json', 'r', encoding='utf-8') as f:
    players_data_t1 = json.load(f)

with open('P_TeamStanding23_24.json', 'r', encoding='utf-8') as f:
    team_data_p_23 = json.load(f)

with open('T1_TeamStanding23_24.json', 'r', encoding='utf-8') as f:
    team_data_t1_23 = json.load(f)

with open('P_TeamStanding24_25.json', 'r', encoding='utf-8') as f:
    team_data_p_24 = json.load(f)

with open('T1_TeamSteanding24_25.json', 'r', encoding='utf-8') as f:
    team_data_t1_24 = json.load(f)

# 解析戰績資料
def parse_team_standings(team_data):
    team_standings = {}
    for team in team_data:
        team_standings[team['team_name']] = {
            "wins": team['wins'],
            "losses": team['losses'],
            "pct": float(team['pct'].replace('%', '')) / 100
        }
    return team_standings

team_standings_p_23 = parse_team_standings(team_data_p_23)
team_standings_t1_23 = parse_team_standings(team_data_t1_23)
team_standings_p_24 = parse_team_standings(team_data_p_24)
team_standings_t1_24 = parse_team_standings(team_data_t1_24)

# 合併所有聯盟的戰績資料
all_team_standings = {**team_standings_p_23, **team_standings_t1_23, **team_standings_p_24, **team_standings_t1_24}

# 球隊名稱映射表(改名時要更新)
team_name_mapping = {
    "高雄鋼鐵人": "高雄17直播鋼鐵人",
    "台啤永豐雲豹": "桃園台啤永豐雲豹",
    "臺北戰神": "臺北台新戰神",
    "台鋼獵鷹": "臺南台鋼獵鷹",
    "夢想家": "福爾摩沙夢想家",
    "攻城獅": "新竹御頂攻城獅",
    "領航猿": "桃園璞園領航猿",
    "富邦勇士": "臺北富邦勇士"
}

# 建表
available_team_names = list(all_team_standings.keys())

# 加入相似名稱推薦避免錯字
def get_mapped_team_name(team_name):
    mapped_name = team_name_mapping.get(team_name.strip(), team_name.strip())
    if mapped_name not in all_team_standings:
        close_matches = get_close_matches(mapped_name, available_team_names, n=1, cutoff=0.6)
        if close_matches:
            print(f"名稱 '{mapped_name}' 找不到，可能您指的是：'{close_matches[0]}'")
            return close_matches[0]
        else:
            print(f"名稱 '{mapped_name}' 找不到，請檢查輸入是否正確。")
            return None
    return mapped_name

# 處理聯盟數據的通用函數
def process_league(players_data, team_standings):
    teams_data = {}
    for player_data in players_data:
        team = player_data["team"]
        team_name = get_mapped_team_name(team)
        if team_name not in team_standings:
            continue

        # 使用球員得分計算整體實力
        points = float(player_data.get("points", 0))
        rebounds = float(player_data.get("rebounds", 0))
        assists = float(player_data.get("assists", 0))

        # 實力計算公式
        player_score = points * 0.5 + rebounds * 0.3 + assists * 0.2
        pct = team_standings[team_name]['pct']
        overall_score = player_score * pct

        if team_name not in teams_data:
            teams_data[team_name] = overall_score
        else:
            teams_data[team_name] += overall_score

    return teams_data

# 計算 P+ 和 T1 聯盟的球隊實力
p_league_team_strengths = process_league(players_data_p, team_standings_p_23)
t1_league_team_strengths = process_league(players_data_t1, team_standings_t1_23)

# 合併兩個聯盟的球隊實力
all_team_strengths = {**p_league_team_strengths, **t1_league_team_strengths}

# 計算勝率的 Sigmoid 函數
def calculate_win_probability(strength_a, strength_b):
    strength_diff = strength_a - strength_b
    probability = 1 / (1 + np.exp(-strength_diff / 1000))
    return probability * 100

# 根據進攻與防守實力模擬比分
def simulate_score(offensive_strength, defensive_strength):
    avg_points_per_game = 90
    strength_diff = offensive_strength - defensive_strength
    
    # 基本預測分數
    base_score = avg_points_per_game + (strength_diff / 1000) * avg_points_per_game

    # 設定浮動範圍（標準差）
    std_dev = 5  # 標準差為 5 分

    # 計算最小和最大分數
    min_score = round(base_score - std_dev)
    max_score = round(base_score + std_dev)

    # 確保分數不為負數
    min_score = max(0, min_score)
    max_score = max(0, max_score)
    return min_score, max_score


# 模擬比賽
def simulate_match(team_a, team_b, is_a_home):
    team_a = get_mapped_team_name(team_a)
    team_b = get_mapped_team_name(team_b)

    if not team_a or not team_b:
        print(f"球隊名稱輸入錯誤：{team_a} 或 {team_b} 不存在。")
        return

    strength_a = all_team_strengths.get(team_a, 0)
    strength_b = all_team_strengths.get(team_b, 0)

    if is_a_home:
        win_probability_a = calculate_win_probability(strength_a * 1.05, strength_b)
        win_probability_b = 100 - win_probability_a
        score_a_min, score_a_max = simulate_score(strength_a * 1.05, strength_b)
        score_b_min, score_b_max = simulate_score(strength_b, strength_a * 1.05)
    else:
        win_probability_b = calculate_win_probability(strength_b * 1.05, strength_a)
        win_probability_a = 100 - win_probability_b
        score_a_min, score_a_max = simulate_score(strength_a, strength_b * 1.05)
        score_b_min, score_b_max = simulate_score(strength_b * 1.05, strength_a)

    print(f"{team_a} 對 {team_b} 的比賽:")
    if is_a_home:
        print(f"{team_a}（主場）勝率: {win_probability_a:.2f}%, {team_b}（客場）勝率: {win_probability_b:.2f}%")
        print(f"預期比分: {team_a} {score_a_min} ~ {score_a_max} - {team_b} {score_b_min} ~ {score_b_max}")
    else:
        print(f"{team_b}（主場）勝率: {win_probability_b:.2f}%, {team_a}（客場）勝率: {win_probability_a:.2f}%")
        print(f"預期比分: {team_a} {score_a_min} ~ {score_a_max} - {team_b} {score_b_min} ~ {score_b_max}")


team_a = input("請輸入隊伍A名稱: ")
team_b = input("請輸入隊伍B名稱: ")
is_a_home = input("隊伍A是否是主場? (yes/no): ").strip().lower() == "yes"

simulate_match(team_a, team_b, is_a_home)