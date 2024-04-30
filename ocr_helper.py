import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from paddleocr import paddleocr
from paddleocr.paddleocr import (BASE_DIR, SUPPORT_DET_MODEL,
                                 SUPPORT_OCR_MODEL_VERSION, SUPPORT_REC_MODEL,
                                 alpha_to_color, binarize_img, check_gpu,
                                 check_img, confirm_model_dir_url,
                                 get_model_config, logger, maybe_download,
                                 parse_args, parse_lang, predict_system)
from paddleocr import PaddleOCR as RawPaddleOCR

import easyocr


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)
        if params.det_lang is not None:
            det_lang = params.det_lang

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
        if params.ocr_version in ['PP-OCRv3', 'PP-OCRv4']:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(paddleocr.__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self, img, det=True, rec=True, cls=True, bin=False, inv=False, alpha_color=(255, 255, 255)):
        """
        OCR with PaddleOCR
        args:
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det is True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls is True and self.use_angle_cls is False:
            logger.warning(
                'Since the angle classifier is not initialized, it will not be used during the forward process'
            )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if not dt_boxes:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


class EasyOCR:
    def __init__(self, **kwargs):
        self.reader = easyocr.Reader(['en'],
                                     # recog_network='english_g2',
                                     )

    def ocr(self, img, **kwargs):
        default_kwargs = dict(decoder='greedy', beamWidth=5, batch_size=1,
                              workers=0, allowlist='0123456789 :-', blocklist=None, detail=1,
                              rotation_info=None, paragraph=False, min_size=20,
                              contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
                              text_threshold=0.7, low_text=0.4, link_threshold=0.4,
                              canvas_size=2560, mag_ratio=1.,
                              slope_ths=0.03, ycenter_ths=0.5, height_ths=0.5,
                              width_ths=0.2, y_ths=0.5, x_ths=1.0, add_margin=0.1,
                              threshold=0.3, bbox_min_score=0.2, bbox_min_size=3,
                              max_candidates=0,
                              output_format='standard')
        new_kwargs = {k: kwargs.get(k, default_kwargs[k]) for k in default_kwargs}

        out = self.reader.readtext(img, **new_kwargs)
        return out


_paddle_ocr = None
_easy_ocr = None


def ocr_model(model="PaddleOCR"):
    global _paddle_ocr, _easy_ocr
    if model == 'EasyOCR':
        _easy_ocr = _easy_ocr if _easy_ocr is not None else EasyOCR()
        model_instance = _easy_ocr
    else:
        _paddle_ocr = _paddle_ocr if _paddle_ocr is not None else RawPaddleOCR(cls=False, lang="ch", det_lang="ch")
        model_instance = _paddle_ocr
    return model_instance


def recognize(img, model='PaddleOCR', **kwargs):
    if model != 'EasyOCR':
        kwargs['cls'] = kwargs.get('cls', False)

    ocr_inst = ocr_model(model)
    out = ocr_inst.ocr(img, **kwargs)

    if model == 'EasyOCR':
        out = [(np.array(pts, np.uint16), text, conf) for pts, text, conf in out]
    else:
        out = [(np.array(pts, np.uint16), text, conf) for pts, (text, conf) in out[0]]
    return out