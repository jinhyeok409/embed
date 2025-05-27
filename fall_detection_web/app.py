from flask import Flask, request, render_template
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
IMAGE_FOLDER = os.path.join('static', 'images')
DB_PATH = 'fall_logs.db'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# 로그 저장 함수
def save_log(timestamp, image_filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, image_path TEXT)")
    cursor.execute("INSERT INTO logs (timestamp, image_path) VALUES (?, ?)", (timestamp, image_filename))
    conn.commit()
    conn.close()

# 로그 조회 함수
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, image_path TEXT)")
    cursor.execute("SELECT timestamp, image_path FROM logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs

# 낙상 감지 로그 수신 API
@app.route('/log_fall', methods=['POST'])
def log_fall():
    timestamp = request.form.get('timestamp')
    image = request.files.get('image')

    if not timestamp or not image:
        return {"error": "Missing timestamp or image"}, 400

    # 이미지 저장
    image_filename = f"fall_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    image.save(image_path)

    # 로그 저장
    save_log(timestamp, image_filename)

    return {"message": "Fall logged successfully"}, 200

# 웹에서 로그 확인 페이지
@app.route('/logs')
def show_logs():
    logs = get_logs()
    return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
