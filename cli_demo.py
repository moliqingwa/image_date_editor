from datetime import datetime
from PIL import Image

from core import modify, DATE_TIME_FORMAT

if __name__ == "__main__":
    # 输入示例
    image_src_path = "input.png"
    image_target_path = "output.png"
    new_date = datetime.strptime("2000/12/25 12:32", "%Y/%m/%d %H:%M")

    new_date_str = new_date.strftime(DATE_TIME_FORMAT)

    image = Image.open(image_src_path)
    modify(image, image_target_path, new_date_str)

