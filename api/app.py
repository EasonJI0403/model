from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# 將 ML 資料夾加入到 Python 路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import predict_game_result  # 導入預測函數

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Basketball Prediction API"})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        team1 = data.get('team1')
        team2 = data.get('team2')
        home_team = data.get('home_team')
        
        if not all([team1, team2, home_team]):
            return jsonify({'error': '缺少必要參數'}), 400
            
        # 調用你的預測函數
        result = predict_game_result(team1, team2, home_team)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()