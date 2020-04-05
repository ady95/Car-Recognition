import os
import cv2
import numpy as np


# 한글 경로 파일 읽기
def imread(filepath, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filepath, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

# 한글 경로 파일 읽기
def imwrite(filepath, img, params=None):
    try:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filepath, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False