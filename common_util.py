import os
import json
from datetime import datetime

def get_now_timestring():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

# 텍스트 읽기
def read_text(file_path):
    text = None
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read()

    return text

# 텍스트 쓰기
def save_text(file_path, text):
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(text)


def save_json(json_path, data):
    save_text(json_path, json.dumps(data, ensure_ascii=False, indent=4))

def load_json(json_path):
    return json.loads(read_text(json_path))

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



if __name__ == "__main__":
    # data = [1,2,3]
    # save_json("data1.json", data)
    data = load_json("data1.json")
    print(data)