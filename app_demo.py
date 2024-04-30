import asyncio
import os
import sys
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool, QObject, QRunnable, QFile
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar, QStatusBar, QFileDialog, QLineEdit, QMessageBox
from loguru import logger

from PyQtImageViewer.QtImageViewer import QtImageViewer
from core import inpaint_model, get_datetime_text, image_inpaint, update_datetime_on_image, modify_file_datetime

TASK_INIT = '初始化'
TASK_PARSE_DATE = '日期解析'
TASK_INPAINT = '局部重绘'
TASK_UPDATE_DATE = '更新日期'
TASK_SAVE = '保存文件'


class TaskItem:
    def __init__(self, raw_img=None):
        self.id = uuid.uuid4().hex
        self.task = TASK_INIT  # INIT, PARSE_DATE, INPAINT, UPDATE_DATE
        self.raw_img: Optional[np.ndarray] = raw_img
        self.date_pts, self.date_text = None, None  # PARSE_DATE 输出
        self.inpaint_img: Optional[np.ndarray] = None  # INPAINT 输出

        self.updated_text: str = ""
        self.updated_img: Optional[np.ndarray] = None  # UPDATE_DATE 输出

        self.kwargs = {}

    def copy(self) -> 'TaskItem':
        other = self

        new_task = TaskItem()
        new_task.id = other.id
        new_task.task = other.task
        new_task.raw_img = other.raw_img if other.raw_img is not None else None
        new_task.date_pts = other.date_pts[:] if other.date_pts is not None else None
        new_task.date_text = other.date_text
        new_task.inpaint_img = other.inpaint_img if other.inpaint_img is not None else None

        new_task.updated_text = other.updated_text
        new_task.updated_img = other.updated_img if other.updated_img is not None else None

        new_task.kwargs = self.kwargs.copy()
        return new_task

    def reset(self):
        self.id = uuid.uuid4().hex
        self.task = TASK_INIT
        self.raw_img = None
        self.date_pts, self.date_text = None, None
        self.inpaint_img = None

        self.updated_text = ""
        self.updated_img = None

        self.kwargs.clear()

    @property
    def message(self):
        return f"写入成功: {self.kwargs['save_path']}" if self.task == TASK_SAVE else self.task

    def __str__(self):
        text = f"id={self.id}, task={self.task}"
        if self.raw_img is not None:
            text += f", raw_img={self.raw_img.shape}"
        if self.date_pts is not None:
            text += f", positions={self.date_pts.tolist()}, text={self.date_text}"
        if self.inpaint_img is not None:
            text += f", inpaint_img={self.inpaint_img.shape}"

        text += f", updated_text={self.updated_text}"
        if self.updated_img is not None:
            text += f", updated_img={self.updated_img.shape}"

        if self.kwargs:
            kvs = [f"{k}={v}" for k, v in self.kwargs.items()]
            text += f", kwargs={{{','.join(kvs)}}}"
        return text

    def __repr__(self):
        return self.__str__()


class WorkerSignals(QObject):
    progress = pyqtSignal(TaskItem)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, parent: 'MainWindow', task: TaskItem):
        super().__init__()
        self.signals = WorkerSignals()
        self.parent = parent
        self.task = task.copy()

    def run(self):
        task = self.task
        logger.info(f"[{task.id}] 任务开始: {self.task}")

        try:
            if task.task == TASK_SAVE:
                self._save(task)
            else:
                self._parse_and_inpaint(task)

            logger.info(f"[{task.id}] 任务结束")
        except Exception as ex:
            logger.error(f"[{task.id}] 任务异常")
            logger.exception(ex)
            print(ex)

    def _save(self, task):
        file_path = task.kwargs["save_path"]
        cv2.imwrite(file_path, cv2.cvtColor(self.task.updated_img, cv2.COLOR_RGB2BGR))
        modify_file_datetime(file_path, self.task.updated_text)

        self.signals.progress.emit(task.copy())

    def _parse_and_inpaint(self, task):
        task.task = TASK_INIT
        self.signals.progress.emit(task.copy())

        logger.info(f"[{task.id}] raw_img={'exists' if task.raw_img is not None else None}")
        if task.raw_img is None:
            return

        rgb_img = task.raw_img

        # 获取图片上的日期文字和位置
        target_pts, target_text = get_datetime_text(rgb_img)
        logger.info(f"[{task.id}] 日期解析结果: {target_pts}, {target_text}")

        task.task, task.date_pts, task.date_text = TASK_PARSE_DATE, target_pts, target_text
        self.signals.progress.emit(task.copy())

        # 图像inpaint
        bgr_img_array = image_inpaint(inpaint_model(), rgb_img, target_pts)
        task.task, task.inpaint_img = TASK_INPAINT, cv2.cvtColor(bgr_img_array, cv2.COLOR_BGR2RGB)
        self.signals.progress.emit(task.copy())
        logger.info(f"[{task.id}] 图片inpaint结束")


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Editor')
        self.setWindowIcon(QIcon('./assets/editor.png'))
        self.setGeometry(100, 100, 1200, 1000)

        # self.image_label = QLabel('显示图像')
        # self.image_label.setStyleSheet('border: 1px solid black;')  # 给标签设置黑色边框
        # # self.image_label.setScaledContents(True)  # 图像自适应大小
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 让标签要显示的内容居中
        # self.image_label.setMinimumSize(640, 480)  # 宽和高保持和摄像头获取的默认大小一致
        #
        # self.setCentralWidget(self.image_label)

        # Create an image viewer widget.
        self.viewer = QtImageViewer()

        # Set viewer's aspect ratio mode.
        # !!! ONLY applies to full image view.
        # !!! Aspect ratio always ignored when zoomed.
        #   Qt.AspectRatioMode.IgnoreAspectRatio: Fit to viewport.
        #   Qt.AspectRatioMode.KeepAspectRatio: Fit in viewport using aspect ratio.
        #   Qt.AspectRatioMode.KeepAspectRatioByExpanding: Fill viewport using aspect ratio.
        self.viewer.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        # Set the viewer's scroll bar behaviour.
        #   Qt.ScrollBarPolicy.ScrollBarAlwaysOff: Never show scroll bar.
        #   Qt.ScrollBarPolicy.ScrollBarAlwaysOn: Always show scroll bar.
        #   Qt.ScrollBarPolicy.ScrollBarAsNeeded: Show scroll bar only when zoomed.
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Allow zooming by draggin a zoom box with the left mouse button.
        # !!! This will still emit a leftMouseButtonReleased signal if no dragging occured,
        #     so you can still handle left mouse button clicks in this way.
        #     If you absolutely need to handle a left click upon press, then
        #     either disable region zooming or set it to the middle or right button.
        self.viewer.regionZoomButton = Qt.MouseButton.LeftButton  # set to None to disable

        # Pop end of zoom stack (double click clears zoom stack).
        self.viewer.zoomOutButton = Qt.MouseButton.RightButton  # set to None to disable

        # Mouse wheel zooming.
        self.viewer.wheelZoomFactor = 1.25  # Set to None or 1 to disable

        # Allow panning with the middle mouse button.
        self.viewer.panButton = Qt.MouseButton.MiddleButton  # set to None to disable
        self.setCentralWidget(self.viewer)

        # setting menu
        open_action = QAction(QIcon('./assets/open.png'), '打开图像', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)

        self.edit_date_text = QLineEdit()
        self.edit_date_text.setText("日期未解析到")

        self.modify_action = QAction(QIcon('./assets/modify.png'), '更新日期', self)
        self.modify_action.setShortcut('Ctrl+M')
        self.modify_action.setEnabled(False)
        self.modify_action.triggered.connect(self.modify)

        self.save_action = QAction(QIcon('./assets/save.png'), '保存图像', self)
        self.save_action.setShortcut('Ctrl+S')
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save)

        undo_action = QAction(QIcon('./assets/undo.png'), '取消', self)
        undo_action.setShortcut('Ctrl+Z')
        # undo_action.triggered.connect(self.image_label.undo)

        redo_action = QAction(QIcon('./assets/redo.png'), '重做', self)
        redo_action.setShortcut('Ctrl+Y')
        # redo_action.triggered.connect(self.text_edit.redo)

        # adding a toolbar
        toolbar = QToolBar('Main toolbar')
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        toolbar.addWidget(self.edit_date_text)
        toolbar.addAction(self.modify_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()

        # status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('状态栏')

        self.filepath: Optional[Path] = None
        self.raw_img = None
        self.task = TaskItem()

        self.threadpool = QThreadPool()

        # worker = Worker(self, self.task)
        # worker.setAutoDelete(True)
        # worker.signals.progress.connect(self.task_progress)
        # self.threadpool.start(worker)

        self.center_on_screen()
        self.show()

    def center_on_screen(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def open_image(self):
        filepath, ok = QFileDialog.getOpenFileName(
            self,
            "打开图像",
            "",
            "Images (*.png *.xpm *.jpg *.bmp)"
        )
        if filepath:
            self.modify_action.setEnabled(False)
            self.save_action.setEnabled(False)

            self.filepath = Path(filepath)
            self.edit_date_text.clear()

            bgr_img = cv2.imread(filepath)
            self.raw_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            self.viewer.setImage(self.raw_img, swap_rgb=False)

            # 如果有旧任务正在执行，中断它
            self.threadpool.clear()

            # 创建新的任务并将其添加到线程池中
            self.task.reset()
            self.task.raw_img = self.raw_img

            self.status_bar.showMessage(f"开始 任务")
            worker = Worker(self, self.task)
            worker.setAutoDelete(True)
            worker.signals.progress.connect(self.task_progress)
            self.threadpool.start(worker)

    def modify(self):
        if self.task.inpaint_img is None:
            return

        new_date_str = self.edit_date_text.text()
        text_position = self.task.date_pts.mean(0).astype(np.uint16).tolist()
        new_img_array = update_datetime_on_image(self.task.inpaint_img, new_date_str, text_position, text_anchor='mm')

        self.task.updated_text = new_date_str
        self.task.updated_img = new_img_array

        self.viewer.setImage(self.task.updated_img, swap_rgb=False)
        self.save_action.setEnabled(True)
        self.status_bar.showMessage("日期修改成功!")

    def save(self):
        if self.task.updated_img is None:
            return

        file_dialog = QFileDialog(self)
        file_dialog.setDefaultSuffix(f"{self.filepath.suffix[1:]}")
        file_path, _ = file_dialog.getSaveFileName(self, "目标图像另存为", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
        if file_path:
            # 检查文件是否重复
            if QFile(file_path).exists():
                os.remove(file_path)

            self.task.kwargs['save_path'] = file_path
            self.task.task = TASK_SAVE
            self.status_bar.showMessage(f"开始 {self.task.task}")

            worker = Worker(self, self.task)
            worker.setAutoDelete(True)
            worker.signals.progress.connect(self.task_progress)
            self.threadpool.start(worker)

    def task_progress(self, task: TaskItem):
        if task.id == self.task.id:
            self.task = task.copy()

            if self.task.task == TASK_PARSE_DATE:
                self.edit_date_text.setText(task.date_text)

                img = cv2.rectangle(self.task.raw_img.copy(),
                                    self.task.date_pts[0].tolist(), self.task.date_pts[2].tolist(),
                                    color=(255, 0, 0), thickness=5,
                                    )
                self.viewer.setImage(img, swap_rgb=False)
            elif self.task.task == TASK_INPAINT:
                self.viewer.setImage(self.task.inpaint_img, swap_rgb=False)
                self.modify_action.setEnabled(True)

            self.status_bar.showMessage(f"{task.message} 完成")

    def closeEvent(self, event):
        # 执行清理操作
        self.threadpool.clear()

        event.accept()  # 接受关闭事件，退出应用程序


if __name__ == '__main__':
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
    logger.add("logs/log.log", encoding="utf-8", compression="zip", enqueue=True,
               rotation="10MB", retention="7 days", level="INFO")

    app = QApplication(sys.argv)

    window = MainWindow()
    sys.exit(app.exec())
