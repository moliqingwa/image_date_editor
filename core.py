import math
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest, HDStrategy
from pywintypes import Time  # 可以忽视这个 Time 报错（运行程序还是没问题的）
from win32file import CreateFile, SetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING

import ocr_helper

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 时间格式

RE_DATE_TIME = re.compile("^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<hour>[01]?[0-9]|2[0-3])\s?:\s?(?P<minute>[0-5][0-9])\s?:\s?(?P<second>[0-5][0-9])$")
RE_DATE_TIME_DIM = re.compile("^(?P<date>\d{4}-\d{2}-\d{2})[\s\W]+(?P<hour>[01]?[0-9]|2[0-3])\s?[:|]\s?(?P<minute>[0-5][0-9])\s?[:|]\s?(?P<second>[0-5][0-9])$")


def time_offset_and_struct(times, format, offset):
    return time.localtime(time.mktime(time.strptime(times, format)) + offset)


def modify_file_datetime(file_path, file_datetime):
    """
    修改已存在文件的创建日期属性
    """
    # 进行时间偏移 1S，避免创建时间，修改时间，访问时间都一样
    create_time_t = time_offset_and_struct(file_datetime, DATE_TIME_FORMAT, 0)
    # modify_time_t = time_offset_and_struct(file_datetime, FORMAT, 1)
    # access_time_t = time_offset_and_struct(file_datetime, FORMAT, 2)

    fh = CreateFile(file_path, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)

    # create_time, accessTimes, modifyTimes = GetFileTime(fh)
    create_time = Time(time.mktime(create_time_t))
    # access_time = Time(time.mktime(access_time_t))
    # modify_time = Time(time.mktime(modify_time_t))
    SetFileTime(fh, create_time, create_time, create_time)
    CloseHandle(fh)


def get_datetime_text(image_array):
    h, w, _ = image_array.shape
    h_2, w_2 = 2**int(math.log(h // 2, 2)), 2**int(math.log(w // 2, 2))
    xy_bias = np.array([[w - w_2, h - h_2]]*4, dtype=np.uint16)

    new_image_array = image_array[-h_2:, -w_2:]  # 取右下角图片
    text_infos = ocr_helper.recognize(new_image_array)
    # xy_bias = np.array([[w_2, h_2]]*4, dtype=np.uint16)
    text_infos = list(map(lambda x: (x[0] + xy_bias, x[1], x[2]), text_infos))

    text_infos.sort(key=lambda x: tuple(x[0][0].tolist()))  # 先按x升序,再按y升序
    target_pts, target_text = None, None
    for pts, text, conf in text_infos:
        m = RE_DATE_TIME_DIM.match(text)  # 模糊匹配
        if m:
            target_pts, target_text = pts, f"{m.group('date')} {m.group('hour')}:{m.group('minute')}:{m.group('second')}"
            break

    if not target_text:
        raise Exception("未查找到日期数字!")
    return target_pts, target_text


def image_inpaint(inpaint_model, image_array, target_pts):
    x_min, x_max, y_min, y_max = target_pts[:, 0].min(), target_pts[:, 0].max(), target_pts[:, 1].min(), target_pts[:, 1].max()

    image_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    image_mask[y_min: y_max+1, x_min: x_max+1] = 255
    # cv2.imwrite(f"image_mask.jpg", image_mask)

    inpaint_config = InpaintRequest(hd_strategy=HDStrategy.CROP,
                                    hd_strategy_crop_margin=128,
                                    sd_keep_unmasked_area=True,
                                    )
    bgr_image_array = inpaint_model(image_array, image_mask, inpaint_config).astype(np.uint8)
    return bgr_image_array


def update_datetime_on_image(image_array, datetime_str, position, text_anchor=None):
    image = Image.fromarray(image_array, 'RGB')
    image_draw = ImageDraw.Draw(image)

    font_path = "clacon2.ttf"
    font = ImageFont.truetype(font_path, size=57, layout_engine=ImageFont.Layout.BASIC)

    text = datetime_str
    text_color = (255, 255, 255)  # 设置文本颜色(白色)
    image_draw.text(position, text, font=font, fill=text_color, anchor=text_anchor)

    # 小时和冒号 为黑色
    m = re.match("(.*\s\d{2}:)(\d{2}:)(\d{2})$", datetime_str)
    prefix_str, black_str, suffix_str = m.group(1), m.group(2), m.group(3)

    left, top, right, bottom = image_draw.textbbox(position, text, font=font, anchor=text_anchor)  # 全部
    left1, top1, right1, bottom1 = image_draw.textbbox((left, top), prefix_str, font=font, anchor='la')  # 前缀
    image_draw.text((right1, top), black_str, font=font, fill=(0, 0, 0), anchor='la')
    left2, top2, right2, bottom2 = image_draw.textbbox(position, black_str, font=font, anchor=text_anchor)  # 中间(黑色)
    left3, top3, right3, bottom3 = image_draw.textbbox(position, suffix_str, font=font, anchor=text_anchor)  # 后缀

    return np.array(image, np.uint8)


_inpaint_model = None


def inpaint_model():
    global _inpaint_model
    if _inpaint_model is None:
        _inpaint_model = LaMa(device='cpu')
    return _inpaint_model


def modify(rgb_img, img_target_path, new_date_str):
    if isinstance(rgb_img, np.ndarray):
        img_rgb_array = rgb_img.copy()
    else:
        img_rgb_array = np.asarray(rgb_img).copy()  # RGB

    # 获取图片上的日期文字和位置
    target_pts, target_text = get_datetime_text(img_rgb_array)

    bgr_img_array = image_inpaint(inpaint_model(), img_rgb_array, target_pts)
    new_img_array = cv2.cvtColor(bgr_img_array, cv2.COLOR_BGR2RGB)

    # 计算文本位置
    # text_position = (image.width - 875, image.height - 65)
    # text_position = target_pts[0].tolist()
    text_position = target_pts.mean(0).astype(np.uint16).tolist()
    new_img_array = update_datetime_on_image(new_img_array, new_date_str, text_position, text_anchor='mm')

    # 更新文件的创建日期
    cv2.imwrite(img_target_path, cv2.cvtColor(new_img_array, cv2.COLOR_RGB2BGR))
    modify_file_datetime(img_target_path, new_date_str)

    # 返回最新的图像
    return Image.fromarray(new_img_array)
