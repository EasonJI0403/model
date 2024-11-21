import json
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from difflib import get_close_matches

# 讀取數據函數
def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON in {filename}. Skipping.")
        return []

# 讀取球員數據
p_player_data_23 = load_json('P_Players_Performance_23_24.json')
t1_player_data_23 = load_json('T1_Players_performance_23_24.json')
p_player_data_24 = load_json('P_Players_Performance_24_25.json')
t1_player_data_24 = load_json('T1_Players_performance_24_25.json')

# 讀取球隊戰績數據
p_team_data_23 = load_json('P_TeamStanding23_24.json')
t1_team_data_23 = load_json('T1_TeamStanding23_24.json')
p_team_data_24 = load_json('P_TeamStanding24_25.json')
t1_team_data_24 = load_json('T1_TeamStanding24_25.json')

# 讀取球隊季節性表現數據
t1_season_data = load_json('T1_Season_teams_performance_24_25.json')
p_season_data = load_json('P_Season_teams_Performance_24_25.json')

# 合併所有數據
all_player_data = p_player_data_23 + t1_player_data_23 + p_player_data_24 + t1_player_data_24
all_team_data = p_team_data_23 + t1_team_data_23 + p_team_data_24 + t1_team_data_24
all_season_data = t1_season_data + p_season_data



# 建立球員位置字典
position_dict = {player_data['player']: player_data['position'] for player_data in all_player_data}

# 球員評分權重
weights = {
    'G': {
        'points': 1.0, 'All_goals_pct': 0.8, 'field_goals_two_pct': 0.6,
        'field_goals_three_pct': 1.0, 'free_throws_pct': 0.7, 'rebounds': 0.5,
        'assists': 1.2, 'steals': 1.0, 'blocks': 0.3, 'turnovers': -0.8, 'fouls': -0.5
    },
    'F': {
        'points': 1.0, 'All_goals_pct': 0.8, 'field_goals_two_pct': 0.7,
        'field_goals_three_pct': 0.6, 'free_throws_pct': 0.6, 'rebounds': 1.0,
        'assists': 0.6, 'steals': 0.7, 'blocks': 0.8, 'turnovers': -0.7, 'fouls': -0.6
    },
    'C': {
        'points': 0.8, 'All_goals_pct': 0.7, 'field_goals_two_pct': 0.8,
        'field_goals_three_pct': 0.3, 'free_throws_pct': 0.5, 'rebounds': 1.2,
        'assists': 0.4, 'steals': 0.4, 'blocks': 1.0, 'turnovers': -0.6, 'fouls': -0.7
    }
}

# 計算球員得分
def calculate_player_score(player_data):
    player_name = player_data['player']
    player_position = position_dict.get(player_name, 'G')
    player_score = 0
    for stat, weight in weights[player_position].items():
        try:
            player_score += weight * float(player_data[stat])
        except (ValueError, KeyError):
            continue
    return player_score

# 計算每支球隊的最強9人
team_top_players = defaultdict(list)
for player_data in all_player_data:
    player_name = player_data['player']
    player_team = player_data['team']
    player_score = calculate_player_score(player_data)
    team_top_players[player_team].append((player_name, player_score))

for team, players in team_top_players.items():
    team_top_players[team] = sorted(players, key=lambda x: x[1], reverse=True)[:9]

# 球隊名稱映射表
team_name_mapping = {
    "領航猿": "桃園璞園領航猿", "勇士": "臺北富邦勇士", "鋼鐵人": "高雄全家海神",
    "獵鷹": "高雄全家海神", "桃園璞園領航猿": "桃園璞園領航猿", "福爾摩沙夢想家": "福爾摩沙夢想家",
    "新北國王": "新北國王", "新竹御嵿攻城獅": "新竹御嵿攻城獅", "臺北富邦勇士": "臺北富邦勇士",
    "高雄全家海神": "高雄全家海神"
}

# 創建球隊平均得分字典
team_avg_points = {team['team']: float(team['points']) for team in all_season_data}

# 準備訓練數據
X = []
y = []
for team_data in all_team_data:
    team_name = team_data['team_name']
    team_name = team_name_mapping.get(team_name, team_name)
    wins = team_data['wins']
    losses = team_data['losses']
    avg_points = team_avg_points.get(team_name, 0)
    top_players = team_top_players.get(team_name, [])
    top_player_scores = [score for _, score in top_players]
    if len(top_player_scores) < 9:
        top_player_scores.extend([0] * (9 - len(top_player_scores)))
    X.append([wins, losses, avg_points] + top_player_scores[:9])
    pct = float(team_data['pct'].rstrip('%')) / 100
    y.append(1 if pct > 0.5 else 0)

X = np.array(X)
y = np.array(y)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練分類模型
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 模糊匹配函數
def find_team(team_name, all_team_data):
    exact_match = next((team for team in all_team_data if team['team_name'] == team_name), None)
    if exact_match:
        return exact_match
    all_names = [team['team_name'] for team in all_team_data]
    close_matches = get_close_matches(team_name, all_names, n=1, cutoff=0.6)
    if close_matches:
        return next(team for team in all_team_data if team['team_name'] == close_matches[0])
    return None

# 預測函數
def predict_game_result(team1, team2, is_home_team):
    print(f"Predicting result for: {team1} vs {team2}")
    print(f"Home team: {is_home_team}")

    team1_name = team_name_mapping.get(team1, team1)
    team2_name = team_name_mapping.get(team2, team2)
    print(f"Mapped team names: {team1_name}, {team2_name}")

    team1_stats = find_team(team1_name, all_team_data)
    team2_stats = find_team(team2_name, all_team_data)

    if not team1_stats or not team2_stats:
        return f"找不到指定的球隊數據: {team1_name if not team1_stats else ''} {team2_name if not team2_stats else ''}"

    team1_wins = team1_stats['wins']
    team1_losses = team1_stats['losses']
    team1_avg_points = team_avg_points.get(team1_name, 0)
    team1_top_players = team_top_players.get(team1_name, [])
    team1_top_player_scores = [score for _, score in team1_top_players]
    if len(team1_top_player_scores) < 9:
        team1_top_player_scores.extend([0] * (9 - len(team1_top_player_scores)))

    team2_wins = team2_stats['wins']
    team2_losses = team2_stats['losses']
    team2_avg_points = team_avg_points.get(team2_name, 0)
    team2_top_players = team_top_players.get(team2_name, [])
    team2_top_player_scores = [score for _, score in team2_top_players]
    if len(team2_top_player_scores) < 9:
        team2_top_player_scores.extend([0] * (9 - len(team2_top_player_scores)))

    X_pred = [[team1_wins, team1_losses, team1_avg_points] + team1_top_player_scores[:9],
              [team2_wins, team2_losses, team2_avg_points] + team2_top_player_scores[:9]]

    X_pred = np.array(X_pred)

    # 預測勝率
    win_probs = clf.predict_proba(X_pred)
    team1_win_prob = win_probs[0][1]
    team2_win_prob = win_probs[1][1]

    # 預測得分
    team1_score_pred = team1_avg_points
    team2_score_pred = team2_avg_points

    # 添加隨機變化
    team1_score_pred += np.random.normal(0, 5)
    team2_score_pred += np.random.normal(0, 5)

    # 應用主場優勢
    if is_home_team == team1_name:
        team1_score_pred *= 1.05
    elif is_home_team == team2_name:
        team2_score_pred *= 1.05

    return f"預測{team1_name}對{team2_name}的比賽結果:\n" \
           f"  {team1_name}勝率: {team1_win_prob:.2f}\n" \
           f"  {team2_name}勝率: {team2_win_prob:.2f}\n" \
           f"  {team1_name}預測得分: {team1_score_pred:.1f}\n" \
           f"  {team2_name}預測得分: {team2_score_pred:.1f}\n" \
           f"  主場隊伍: {is_home_team}"

# 測試預測
print(predict_game_result("臺北富邦勇士", "新北國王", "新北國王"))