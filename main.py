import sys
import json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QCheckBox, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint
from PIL import Image, ImageDraw


class LampScreen:
    gridcolor = (76, 45, 28)
    offcolor = (173, 101, 62)
    oncolor = (242, 201, 159)

    def __init__(self, sx, sy):
        self.sx = sx
        self.sy = sy
        self.pixels = [[False for _ in range(sy)] for _ in range(sx)]
        self.update_image()

    def update_image(self):
        self.image = Image.new("RGB", (self.sx * 10, self.sy * 10), LampScreen.gridcolor)
        draw = ImageDraw.Draw(self.image)
        for x in range(self.sx):
            for y in range(self.sy):
                color = LampScreen.oncolor if self.pixels[x][y] else LampScreen.offcolor
                draw.rectangle((x * 10, y * 10, x * 10 + 8, y * 10 + 8), color)

    def set_pixel(self, x, y, on):
        if 0 <= x < self.sx and 0 <= y < self.sy:
            self.pixels[x][y] = on
            self.update_image()

    def toggle_pixel(self, x, y):
        if 0 <= x < self.sx and 0 <= y < self.sy:
            self.pixels[x][y] = not self.pixels[x][y]
            self.update_image()

    def export_png(self, filename):
        self.image.save(filename, "PNG")

    def export_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.pixels, f)

    def load_json(self, filename):
        with open(filename, "r") as f:
            self.pixels = json.load(f)
        self.update_image()

    def from_image(self, path):
        img = Image.open(path).convert("L").resize((self.sx, self.sy))
        for x in range(self.sx):
            for y in range(self.sy):
                self.pixels[x][y] = img.getpixel((x, y)) > 128
        self.update_image()


class LampApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Redstone lamp editor")
        self.lamp = LampScreen(32, 32)
        self.drawing = True
        self.erasing = False
        self.mouse_down = False

        self.label = QLabel()
        self.label.setPixmap(self.get_pixmap())
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

        # Buttons
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_png.clicked.connect(self.export_png)

        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.clicked.connect(self.export_json)

        self.btn_load_json = QPushButton("Load JSON")
        self.btn_load_json.clicked.connect(self.load_json)

        self.btn_img2lamp = QPushButton("Image → Lamp")
        self.btn_img2lamp.clicked.connect(self.img2lamp)

        self.chk_draw = QCheckBox("Drawing mode")
        self.chk_draw.setChecked(True)
        self.chk_draw.stateChanged.connect(self.toggle_drawing)

        self.chk_erase = QCheckBox("Erase mode")
        self.chk_erase.setChecked(False)
        self.chk_erase.stateChanged.connect(self.toggle_erase)

        # Layout
        button_layout = QHBoxLayout()
        for btn in [self.btn_export_png, self.btn_export_json, self.btn_load_json, self.btn_img2lamp]:
            button_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.chk_draw)
        layout.addWidget(self.chk_erase)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_preview()

    def mouse_press(self, event):
        if not self.drawing:
            return
        self.mouse_down = True
        self.paint_at(event)

    def mouse_move(self, event):
        if self.mouse_down and self.drawing:
            self.paint_at(event)

    def mouse_release(self, event):
        self.mouse_down = False

    def paint_at(self, event):
        pos = event.position().toPoint()
        x = pos.x() // 10
        y = pos.y() // 10
        if self.erasing:
            self.lamp.set_pixel(x, y, False)
        else:
            self.lamp.set_pixel(x, y, True)
        self.update_preview()

    def toggle_drawing(self):
        self.drawing = self.chk_draw.isChecked()

    def toggle_erase(self):
        self.erasing = self.chk_erase.isChecked()

    def update_preview(self):
        self.label.setPixmap(self.get_pixmap())

    def get_pixmap(self):
        self.lamp.update_image()
        data = self.lamp.image.tobytes("raw", "RGB")
        qimg = QImage(data, self.lamp.image.width, self.lamp.image.height, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(320, 320, Qt.AspectRatioMode.KeepAspectRatio)

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Files (*.png)")
        if path:
            self.lamp.export_png(path)
            QMessageBox.information(self, "Success", f"PNG opgeslagen naar:\n{path}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
        if path:
            self.lamp.export_json(path)
            QMessageBox.information(self, "Success", f"JSON saved to:\n{path}")

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load JSON", "", "JSON Files (*.json)")
        if path:
            self.lamp.load_json(path)
            self.update_preview()

    def img2lamp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecteer img", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.lamp.from_image(path)
            self.update_preview()


def main():
    app = QApplication(sys.argv)
    window = LampApp()
    window.resize(400, 480)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
