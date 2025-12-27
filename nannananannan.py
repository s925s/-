# -*- coding: utf-8 -*-
# click_reveal_pro_fixed.py
# 依存: pip install PySide6 Pillow numpy

import sys, os, time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageOps, ImageFilter, PngImagePlugin
from PySide6 import QtCore, QtGui, QtWidgets

os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

# ---------------- 共通ユーティリティ ----------------
def ensure_rgba(img: Image.Image) -> Image.Image:
    """画像をRGBAに変換"""
    return img if img.mode == "RGBA" else img.convert("RGBA")

def fit_into_canvas(img: Image.Image, W: int, H: int, mode: str = "auto") -> Image.Image:
    """画像をキャンバスにフィットさせる"""
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))
    
    if mode == "fit_w":
        s = W / iw
    elif mode == "fit_h":
        s = H / ih
    else:
        s = min(W / iw, H / ih)
    
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    r = img.resize((nw, nh), Image.LANCZOS)
    can = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    can.alpha_composite(ensure_rgba(r), ((W - nw) // 2, (H - nh) // 2))
    return can

def gaussian(img: Image.Image, radius: int) -> Image.Image:
    """画像にガウスぼかしを適用"""
    if radius <= 0 or not hasattr(img, 'filter'):
        return img
    
    try:
        return img.filter(ImageFilter.GaussianBlur(radius))
    except Exception as e:
        print(f"Warning: GaussianBlur failed (radius={radius}): {e}")
        return img

def to_gray01(img: Image.Image) -> np.ndarray:
    """画像を0-1のグレースケール配列に変換"""
    L = ImageOps.grayscale(img)
    return np.asarray(L, dtype=np.float32) / 255.0

def pil_to_qpix(img: Image.Image) -> QtGui.QPixmap:
    """PIL画像をQPixmapに変換"""
    if img is None or not hasattr(img, 'mode'):
        return QtGui.QPixmap()
    
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    
    data = img.tobytes("raw", img.mode)
    if img.mode == "RGB":
        qimg = QtGui.QImage(data, img.width, img.height, 3 * img.width, QtGui.QImage.Format.Format_RGB888)
    else:
        qimg = QtGui.QImage(data, img.width, img.height, 4 * img.width, QtGui.QImage.Format.Format_RGBA8888)
    
    return QtGui.QPixmap.fromImage(qimg)

# -------- 厳密合成パラメータ --------
@dataclass
class ExactParams:
    feather_px: int = 0
    mono_internal: bool = True
    clamp_eps: float = 1e-4
    soft_clip: bool = True

def exact_ab_to_rgba(A: Image.Image, B: Image.Image, p: ExactParams, W: int, H: int,
                     fit_T: str, fit_O: str, bg_white: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """2枚の画像から透明PNGを生成（Androidモード）"""
    A = fit_into_canvas(ensure_rgba(A), W, H, fit_T)
    B = fit_into_canvas(ensure_rgba(B), W, H, fit_O)

    # モノクロ変換
    if p.mono_internal:
        a = to_gray01(A)
        b = to_gray01(B)
        A3 = np.stack([a, a, a], -1)
        B3 = np.stack([b, b, b], -1)
    else:
        A3 = np.asarray(A, dtype=np.float32) / 255.0
        B3 = np.asarray(B, dtype=np.float32) / 255.0
        A3 = A3[..., :3]
        B3 = B3[..., :3]

    # アルファ計算
    bg_white_norm = np.array([c/255.0 for c in bg_white])
    alpha = 1.0 + (B3 - A3) / bg_white_norm
    alpha = np.mean(np.clip(alpha, -10, 10), axis=-1, keepdims=True)
    
    if p.soft_clip:
        alpha = np.where(alpha < p.clamp_eps, p.clamp_eps, np.minimum(alpha, 1.0))
    else:
        alpha = np.clip(alpha, p.clamp_eps, 1.0)

    # 前景色計算
    F = np.clip(B3 / alpha, 0.0, 1.0)
    
    # 合成
    a8 = (np.clip(alpha, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    rgb8 = (F * 255 + 0.5).astype(np.uint8)
    out = np.dstack([rgb8, a8])
    img = Image.fromarray(out, "RGBA")
    
    # ここで gaussian 関数を呼び出す
    if p.feather_px > 0:
        img = gaussian(img, p.feather_px)
    
    return img

# ------------- プレビュー（投げ縄機能） -------------
class Preview(QtWidgets.QWidget):
    def __init__(self, bg_color: QtGui.QColor):
        super().__init__()
        self.bg_color = bg_color
        self.img: Optional[Image.Image] = None
        self.lasso_points: List[QtCore.QPoint] = []
        self.lasso_active = False
        self.lasso_mask: Optional[Image.Image] = None
        self.lasso_enabled = True
        self.is_dragging = False

    def set_lasso_enabled(self, enabled: bool):
        """投げ縄機能の有効/無効を切り替え"""
        self.lasso_enabled = enabled
        if not enabled:
            self.clear_lasso()
        self.update()

    def set_image(self, img: Image.Image):
        """画像を設定"""
        self.img = img
        self.update()

    def get_display_rect(self) -> QtCore.QRect:
        """画像が表示される領域を計算"""
        if self.img is None:
            return QtCore.QRect()
        
        W, H = self.width(), self.height()
        pm = pil_to_qpix(self.img)
        iw, ih = pm.width(), pm.height()
        
        if iw <= 0 or ih <= 0:
            return QtCore.QRect()
        
        s = min(W / iw, H / ih)
        dw, dh = int(iw * s), int(ih * s)
        x_offset = (W - dw) // 2
        y_offset = (H - dh) // 2
        
        return QtCore.QRect(x_offset, y_offset, dw, dh)

    def get_lasso_mask(self) -> Optional[Image.Image]:
        """投げ縄領域のマスクを生成"""
        if self.img is None or len(self.lasso_points) < 3:
            return None
        
        W, H = self.img.size
        mask = Image.new("L", (W, H), 0)
        
        display_rect = self.get_display_rect()
        if display_rect.width() <= 0 or display_rect.height() <= 0:
            return None
        
        # プレビュー座標を画像座標に変換
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        scaled_points = []
        
        for pt in self.lasso_points:
            img_x = (pt.x() - display_rect.left()) / display_rect.width() * W
            img_y = (pt.y() - display_rect.top()) / display_rect.height() * H
            img_x = max(0, min(W-1, img_x))
            img_y = max(0, min(H-1, img_y))
            scaled_points.append((img_x, img_y))
        
        if len(scaled_points) >= 3:
            draw.polygon(scaled_points, fill=255)
        
        return mask

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        """マウス押下時"""
        if not self.lasso_enabled or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        
        self.lasso_points = [ev.position().toPoint()]
        self.lasso_active = True
        self.is_dragging = True
        self.update()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        """マウス移動時"""
        if not self.lasso_enabled or not self.is_dragging:
            return
        
        self.lasso_points.append(ev.position().toPoint())
        self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        """マウス解放時"""
        if not self.lasso_enabled or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        
        self.is_dragging = False
        if len(self.lasso_points) >= 3:
            self.finalize_lasso()

    def finalize_lasso(self):
        """投げ縄を確定"""
        if len(self.lasso_points) > 2:
            self.lasso_mask = self.get_lasso_mask()
            self.lasso_points.clear()
            self.lasso_active = False
            self.update()
            return True
        
        self.clear_lasso()
        return False

    def clear_lasso(self):
        """投げ縄をクリア"""
        self.lasso_points.clear()
        self.lasso_active = False
        self.lasso_mask = None
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        """描画処理"""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        W, H = self.width(), self.height()

        # 背景
        p.fillRect(0, 0, W, H, self.bg_color)

        # 画像
        if self.img is not None:
            display_rect = self.get_display_rect()
            pm = pil_to_qpix(self.img)
            p.drawPixmap(display_rect, pm)

        # ラッソ線
        if self.lasso_enabled and (self.lasso_active or len(self.lasso_points) > 0):
            p.setPen(QtGui.QPen(QtGui.QColor(255, 100, 100), 2))
            
            if len(self.lasso_points) > 1:
                # 線
                for i in range(len(self.lasso_points) - 1):
                    p.drawLine(self.lasso_points[i], self.lasso_points[i + 1])
                
                # 閉じた形状
                if not self.is_dragging and len(self.lasso_points) > 2:
                    p.drawLine(self.lasso_points[-1], self.lasso_points[0])
            
            # ポイント
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 100, 100)))
            for i, pt in enumerate(self.lasso_points):
                radius = 6 if i == 0 else (4 if i == len(self.lasso_points)-1 else 2)
                p.drawEllipse(pt, radius, radius)

# ---------------- メインウィンドウ ----------------
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Click Reveal Pro - Android Mode")
        self.resize(1400, 800)

        # 画像データ
        self.img_thumb: Optional[Image.Image] = None
        self.img_open: Optional[Image.Image] = None

        # 設定
        self.canvas_w = 900
        self.canvas_h = 600
        self.fit_T = "auto"
        self.fit_O = "auto"
        self.ex = ExactParams()

        self._build_ui()
        self._apply_theme()
        self.update_preview()

    def _build_ui(self):
        """UI構築"""
        root = QtWidgets.QWidget()
        L = QtWidgets.QHBoxLayout(root)
        L.setContentsMargins(8, 8, 8, 8)
        L.setSpacing(8)

        # 左側：プレビュー
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 投げ縄設定
        lasso_group = QtWidgets.QGroupBox("投げ縄設定")
        lasso_layout = QtWidgets.QHBoxLayout(lasso_group)
        
        self.combo_lasso_preview = QtWidgets.QComboBox()
        self.combo_lasso_preview.addItems(["上プレビューのみ", "下プレビューのみ", "両方表示"])
        self.combo_lasso_preview.setCurrentIndex(2)
        self.combo_lasso_preview.currentIndexChanged.connect(self.on_lasso_preview_change)
        
        self.combo_lasso_mode = QtWidgets.QComboBox()
        self.combo_lasso_mode.addItems(["選択範囲を保護", "選択範囲を切り抜き"])
        self.combo_lasso_mode.currentIndexChanged.connect(self.update_preview)
        
        self.btn_clear_lasso = QtWidgets.QPushButton("クリア")
        self.btn_clear_lasso.clicked.connect(self.clear_lasso)
        
        lasso_layout.addWidget(QtWidgets.QLabel("表示:"))
        lasso_layout.addWidget(self.combo_lasso_preview)
        lasso_layout.addWidget(QtWidgets.QLabel("モード:"))
        lasso_layout.addWidget(self.combo_lasso_mode)
        lasso_layout.addWidget(self.btn_clear_lasso)
        lasso_layout.addStretch()

        # プレビュー
        self.prev_white = Preview(QtGui.QColor(255, 255, 255))
        self.prev_white.setMinimumSize(400, 300)
        
        self.prev_black = Preview(QtGui.QColor(0, 0, 0))
        self.prev_black.setMinimumSize(400, 300)

        left_layout.addWidget(lasso_group)
        left_layout.addWidget(QtWidgets.QLabel("白背景プレビュー"))
        left_layout.addWidget(self.prev_white, 1)
        left_layout.addWidget(QtWidgets.QLabel("黒背景プレビュー"))
        left_layout.addWidget(self.prev_black, 1)

        # 右側：コントロール
        tabs = QtWidgets.QTabWidget()
        tabs.setFixedWidth(470)
        
        # 基本タブ
        basic_tab = QtWidgets.QWidget()
        basic_layout = QtWidgets.QVBoxLayout(basic_tab)
        basic_layout.setContentsMargins(6, 6, 6, 6)

        # 画像設定
        img_group = QtWidgets.QGroupBox("画像設定")
        img_layout = QtWidgets.QGridLayout(img_group)
        
        self.lblT = QtWidgets.QLabel("サムネ用: 未選択")
        self.lblO = QtWidgets.QLabel("開いたとき: 未選択")
        
        btn_load_thumb = QtWidgets.QPushButton("サムネ用を読み込む")
        btn_load_thumb.clicked.connect(lambda: self.load_img("T"))
        
        btn_load_open = QtWidgets.QPushButton("開いたときを読み込む")
        btn_load_open.clicked.connect(lambda: self.load_img("O"))
        
        btn_swap = QtWidgets.QPushButton("画像を入れ替え")
        btn_swap.clicked.connect(self.swap)

        # フィットモード
        self.comboFitT = QtWidgets.QComboBox()
        self.comboFitT.addItems(["全体フィット", "横幅基準", "縦幅基準"])
        self.comboFitO = QtWidgets.QComboBox()
        self.comboFitO.addItems(["全体フィット", "横幅基準", "縦幅基準"])
        self.comboFitT.currentIndexChanged.connect(self.on_fit_change)
        self.comboFitO.currentIndexChanged.connect(self.on_fit_change)

        img_layout.addWidget(self.lblT, 0, 0, 1, 2)
        img_layout.addWidget(btn_load_thumb, 1, 0)
        img_layout.addWidget(self.comboFitT, 1, 1)
        img_layout.addWidget(self.lblO, 2, 0, 1, 2)
        img_layout.addWidget(btn_load_open, 3, 0)
        img_layout.addWidget(self.comboFitO, 3, 1)
        img_layout.addWidget(btn_swap, 4, 0, 1, 2)

        # サイズ設定
        size_group = QtWidgets.QGroupBox("出力サイズ")
        size_layout = QtWidgets.QGridLayout(size_group)
        
        self.spinW = QtWidgets.QSpinBox()
        self.spinW.setRange(320, 8192)
        self.spinW.setValue(self.canvas_w)
        
        self.spinH = QtWidgets.QSpinBox()
        self.spinH.setRange(320, 8192)
        self.spinH.setValue(self.canvas_h)
        
        self.spinW.valueChanged.connect(self.on_size_change)
        self.spinH.valueChanged.connect(self.on_size_change)
        
        size_layout.addWidget(QtWidgets.QLabel("横幅"), 0, 0)
        size_layout.addWidget(self.spinW, 0, 1)
        size_layout.addWidget(QtWidgets.QLabel("高さ"), 1, 0)
        size_layout.addWidget(self.spinH, 1, 1)

        # 書き出し
        export_group = QtWidgets.QGroupBox("書き出し")
        export_layout = QtWidgets.QGridLayout(export_group)
        
        self.editDir = QtWidgets.QLineEdit(os.path.abspath("."))
        btn_pick_dir = QtWidgets.QPushButton("フォルダ選択")
        btn_pick_dir.clicked.connect(self.pick_dir)
        
        btn_export = QtWidgets.QPushButton("PNGで書き出し")
        btn_export.setFixedHeight(42)
        btn_export.clicked.connect(self.export)
        
        export_layout.addWidget(QtWidgets.QLabel("保存先"), 0, 0)
        export_layout.addWidget(self.editDir, 0, 1)
        export_layout.addWidget(btn_pick_dir, 0, 2)
        export_layout.addWidget(btn_export, 1, 0, 1, 3)

        basic_layout.addWidget(img_group)
        basic_layout.addWidget(size_group)
        basic_layout.addWidget(export_group)
        basic_layout.addStretch(1)
        
        tabs.addTab(basic_tab, "基本設定")

        L.addWidget(left_panel, 1)
        L.addWidget(tabs, 0)

        self.setCentralWidget(root)

    def _apply_theme(self):
        """ダークテーマ適用（堅牢版）"""
        self.setStyleSheet("""
        /* メインウィンドウ */
        QMainWindow { 
            background: #1a1a1a; 
        }
        
        /* 基本ウィジェット（明示的に白文字） */
        QWidget { 
            background: #1a1a1a; 
            color: #ffffff; 
            font-size: 13px; 
            font-family: 'Segoe UI', 'Meiryo', sans-serif;
        }
        
        /* グループボックス */
        QGroupBox { 
            border: 1px solid #444; 
            border-radius: 8px; 
            margin-top: 8px; 
            padding-top: 12px;
            background: #2a2a2a;
        }
        
        QGroupBox::title {
            color: #ffffff;  /* 明示的に白文字 */
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px;
            background: #2a2a2a;
        }
        
        /* ボタン */
        QPushButton { 
            background: #ffffff; 
            color: #000000; 
            border: none;
            border-radius: 6px; 
            padding: 8px 12px;
            font-weight: bold;
        }
        
        QPushButton:hover { 
            background: #e0e0e0; 
        }
        
        QPushButton:pressed {
            background: #c0c0c0;
        }
        
        QPushButton:disabled {
            background: #555;
            color: #888;
        }
        
        /* 入力系（明示的に白文字） */
        QLineEdit, QSpinBox, QComboBox { 
            border: 1px solid #444; 
            border-radius: 6px; 
            padding: 6px;
            background: #2a2a2a;
            color: #ffffff;
            selection-background-color: #4a4a4a;
            selection-color: #ffffff;
        }
        
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #666;
        }
        
        /* コンボボックスのドロップダウン */
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
            margin-right: 6px;
        }
        
        /* コンボボックスのポップアップ */
        QComboBox QAbstractItemView {
            background: #2a2a2a;
            color: #ffffff;
            border: 1px solid #444;
            selection-background-color: #4a4a4a;
        }
        
        /* タブ */
        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 8px;
            background: #2a2a2a;
            top: -1px;
        }
        
        QTabBar::tab {
            background: #2a2a2a;
            color: #ffffff;  /* 明示的に白文字 */
            padding: 8px 16px;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: #3a3a3a;
            border-bottom: 1px solid #3a3a3a;
        }
        
        QTabBar::tab:hover {
            background: #333;
        }
        
        /* ラベル（明示的に白文字） */
        QLabel {
            color: #ffffff !important;
            background: transparent;
        }
        
        /* チェックボックス（明示的に白文字） */
        QCheckBox {
            color: #ffffff;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 2px solid #666;
            background: #2a2a2a;
        }
        
        QCheckBox::indicator:checked {
            background: #ffffff;
            border: 2px solid #ffffff;
        }
        
        /* ラジオボタン（明示的に白文字） */
        QRadioButton {
            color: #ffffff;
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 2px solid #666;
            background: #2a2a2a;
        }
        
        QRadioButton::indicator:checked {
            background: #ffffff;
            border: 2px solid #ffffff;
        }
        
        /* スライダー */
        QSlider::groove:horizontal { 
            height: 6px; 
            background: #444; 
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal { 
            width: 16px; 
            background: #ffffff; 
            margin: -5px 0; 
            border-radius: 8px; 
        }
        
        QSlider::handle:horizontal:hover {
            background: #e0e0c0;
        }
        """)

    def on_lasso_preview_change(self):
        """投げ縄プレビュー設定変更"""
        idx = self.combo_lasso_preview.currentIndex()
        self.prev_white.set_lasso_enabled(idx in [0, 2])
        self.prev_black.set_lasso_enabled(idx in [1, 2])

    def swap(self):
        """画像入れ替え"""
        self.img_thumb, self.img_open = self.img_open, self.img_thumb
        t, o = self.lblT.text(), self.lblO.text()
        self.lblT.setText(o)
        self.lblO.setText(t)
        self.update_preview()

    def clear_lasso(self):
        """投げ縄クリア"""
        self.prev_white.clear_lasso()
        self.prev_black.clear_lasso()
        self.update_preview()

    def apply_lasso_to_images(self, white_img: Image.Image, black_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """投げ縄マスクを適用"""
        mask = self.prev_white.lasso_mask or self.prev_black.lasso_mask
        if mask is None:
            return white_img, black_img
        
        # マスクサイズ調整
        if mask.size != white_img.size:
            mask = mask.resize(white_img.size, Image.LANCZOS)
        
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_array = mask_array[..., None]
        
        mode = self.combo_lasso_mode.currentIndex()  # 0:保護, 1:切り抜き
        
        if mode == 0:
            # 選択範囲を保護：黒背景に白画像を重ねる
            black_array = np.array(black_img, dtype=np.float32)
            white_array = np.array(white_img, dtype=np.float32)
            result_black = white_array * mask_array + black_array * (1 - mask_array)
            result_black = np.clip(result_black, 0, 255).astype(np.uint8)
            return white_img, Image.fromarray(result_black, "RGBA")
        else:
            # 選択範囲を切り抜き：白背景に黒画像を重ねる
            white_array = np.array(white_img, dtype=np.float32)
            black_array = np.array(black_img, dtype=np.float32)
            result_white = black_array * mask_array + white_array * (1 - mask_array)
            result_white = np.clip(result_white, 0, 255).astype(np.uint8)
            return Image.fromarray(result_white, "RGBA"), black_img

    def update_preview(self):
        """プレビュー更新"""
        if self.img_thumb is None or self.img_open is None:
            self.prev_white.set_image(None)
            self.prev_black.set_image(None)
            return
        
        # 基本画像生成
        W, H = self.canvas_w, self.canvas_h
        bg_white = (255, 255, 255)
        rgba = exact_ab_to_rgba(self.img_thumb, self.img_open, self.ex, W, H, self.fit_T, self.fit_O, bg_white)
        
        # プレビュー背景
        white_bg = Image.new("RGBA", (W, H), (*bg_white, 255))
        black_bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        
        white_preview = Image.alpha_composite(white_bg, rgba)
        black_preview = Image.alpha_composite(black_bg, rgba)
        
        # 投げ縄適用
        white_final, black_final = self.apply_lasso_to_images(white_preview, black_preview)
        
        self.prev_white.set_image(white_final)
        self.prev_black.set_image(black_final)

    def load_img(self, which: str):
        """画像読み込み"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            f"{'サムネ用' if which=='T' else '開いたとき'}画像を選択",
            "", 
            "画像ファイル (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        
        try:
            img = Image.open(path).convert("RGBA")
            if which == "T":
                self.img_thumb = img
                self.lblT.setText(f"サムネ用: {os.path.basename(path)}")
            else:
                self.img_open = img
                self.lblO.setText(f"開いたとき: {os.path.basename(path)}")
            
            self.update_preview()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "読み込みエラー", str(e))

    def on_fit_change(self):
        """フィットモード変更"""
        idx2mode = {0: "auto", 1: "fit_w", 2: "fit_h"}
        self.fit_T = idx2mode[self.comboFitT.currentIndex()]
        self.fit_O = idx2mode[self.comboFitO.currentIndex()]
        self.update_preview()

    def on_size_change(self):
        """サイズ変更"""
        self.canvas_w = self.spinW.value()
        self.canvas_h = self.spinH.value()
        self.update_preview()

    def pick_dir(self):
        """保存フォルダ選択"""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "保存先フォルダを選択")
        if d:
            self.editDir.setText(d)

    def export(self):
        """PNG書き出し"""
        if self.img_thumb is None or self.img_open is None:
            QtWidgets.QMessageBox.warning(self, "警告", "画像を両方読み込んでください")
            return
        
        W, H = self.canvas_w, self.canvas_h
        out_dir = self.editDir.text()
        
        try:
            # 基本画像
            bg_white = (255, 255, 255)
            rgba = exact_ab_to_rgba(self.img_thumb, self.img_open, self.ex, W, H, self.fit_T, self.fit_O, bg_white)
            
            # 投げ縄マスク適用
            mask = self.prev_white.lasso_mask or self.prev_black.lasso_mask
            if mask is not None:
                white_bg = Image.new("RGBA", (W, H), (*bg_white, 255))
                black_bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                white_preview = Image.alpha_composite(white_bg, rgba)
                black_preview = Image.alpha_composite(black_bg, rgba)
                
                white_final, black_final = self.apply_lasso_to_images(white_preview, black_preview)
                
                # マスク適用後の画像からRGBAを再計算
                final_array = np.array(white_final, dtype=np.float32) / 255.0
                bg_array = np.array(white_bg, dtype=np.float32) / 255.0
                rgb = final_array[..., :3]
                alpha = np.clip(final_array[..., 3:4], 0, 1)
                
                # 背景の影響を除去
                fg_rgb = np.where(alpha > 0.01, rgb / alpha, 0)
                fg_rgb = np.clip(fg_rgb, 0, 1)
                
                rgba_array = np.dstack([fg_rgb, alpha])
                rgba = Image.fromarray((rgba_array * 255).astype(np.uint8), "RGBA")
            
            # メタデータ付き保存
            meta = PngImagePlugin.PngInfo()
            meta.add_text("X-Mode", "厳密合成")
            meta.add_text("X-Canvas", f"{W}x{H}")
            meta.add_text("X-Fit-Thumb", self.fit_T)
            meta.add_text("X-Fit-Open", self.fit_O)
            meta.add_text("X-Lasso-Mode", str(self.combo_lasso_mode.currentIndex()))
            
            out_path = os.path.join(out_dir, f"click-reveal_{time.strftime('%Y%m%d_%H%M%S')}_{W}x{H}.png")
            rgba.save(out_path, "PNG", optimize=True, pnginfo=meta)
            
            QtWidgets.QMessageBox.information(self, "完了", f"保存しました:\n{out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "エラー", f"書き出し失敗:\n{str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec())