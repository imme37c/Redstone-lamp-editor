import sys
import json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QCheckBox, QMessageBox, QSlider, QSpinBox, QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from PIL import Image
import os, shutil, tempfile, zipfile, glob
from pathlib import Path

import numpy as np


def sauvola_threshold(img_gray, window=7, k=0.25):
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape
    mean = np.zeros_like(arr)
    sqmean = np.zeros_like(arr)
    pad = window // 2

    cumsum = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    sq_cumsum = np.cumsum(np.cumsum(arr ** 2, axis=0), axis=1)

    def region_sum(cs, x1, y1, x2, y2):
        return cs[y2, x2] - cs[y1, x2] - cs[y2, x1] + cs[y1, x1]

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            x1, y1 = x - pad, y - pad
            x2, y2 = x + pad, y + pad
            area = (x2 - x1) * (y2 - y1)
            m = region_sum(cumsum, x1, y1, x2, y2) / area
            s = np.sqrt(region_sum(sq_cumsum, x1, y1, x2, y2) / area - m ** 2)
            thresh = m * (1 + k * ((s / 128) - 1))
            mean[y, x] = thresh

    bin_img = (arr > mean).astype(np.uint8) * 255
    return Image.fromarray(bin_img, mode="L")


def smooth_dither(img_gray):
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new = 0 if old < 128 else 255
            err = old - new
            arr[y, x] = new
            if x + 1 < w:
                arr[y, x + 1] += err * 5/16
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3/16
                arr[y + 1, x] += err * 7/16
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * 1/16
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("L")


def floyd_steinberg_dither(img_gray: Image.Image) -> Image.Image:
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new = 0 if old < 128 else 255
            arr[y, x] = new
            err = old - new
            if x + 1 < w:
                arr[y, x+1] += err * 7/16
            if y + 1 < h:
                if x > 0:
                    arr[y+1, x-1] += err * 3/16
                arr[y+1, x] += err * 5/16
                if x + 1 < w:
                    arr[y+1, x+1] += err * 1/16

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("1")


class LampScreen:
    script_dir = Path(__file__).parent
    off_img_path = script_dir / "redstone_lamp.png"
    on_img_path  = script_dir / "redstone_lamp_on.png"

    def __init__(self, sx, sy):
        self.sx = sx
        self.sy = sy
        self.pixels = [[False for _ in range(sy)] for _ in range(sx)]

        self.img_off = Image.open(LampScreen.off_img_path).convert("RGBA")
        self.img_on = Image.open(LampScreen.on_img_path).convert("RGBA")
        self.tile_size = self.img_off.width

        self.wool_textures = {}
        self.wool_avg_colors = {}
        self._load_wools()

        self.image = Image.new("RGBA", (sx * self.tile_size, sy * self.tile_size))
        self.update_image(full=True)

    def _load_wools(self):
        script_dir = Path(__file__).parent
        wools_dir = script_dir / "blocks"
        if not wools_dir.exists():
            return
        for p in sorted(wools_dir.glob("*.png")):
            name = p.stem
            try:
                img = Image.open(p).convert("RGBA")
                if img.width != self.tile_size:
                    img = img.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
                self.wool_textures[name] = img
                arr = np.array(img.convert("RGB")).reshape(-1, 3).astype(np.float32)
                self.wool_avg_colors[name] = arr.mean(axis=0)
            except Exception as e:
                print(f"Failed to load block texture {p}: {e}")

    def update_image(self, full=True, changed_coords=None):
        if full or changed_coords is None:
            for x in range(self.sx):
                for y in range(self.sy):
                    self._paste_tile(x, y)
        else:
            for (x, y) in changed_coords:
                if 0 <= x < self.sx and 0 <= y < self.sy:
                    self._paste_tile(x, y)

    def _paste_tile(self, x, y):
        val = self.pixels[x][y]
        if isinstance(val, str) and val in self.wool_textures:
            tile = self.wool_textures[val]
            self.image.paste(tile, (x * self.tile_size, y * self.tile_size), tile)
        else:
            tile = self.img_on if bool(val) else self.img_off
            self.image.paste(tile, (x * self.tile_size, y * self.tile_size), tile)

    def set_pixel(self, x, y, on_or_name):
        if 0 <= x < self.sx and 0 <= y < self.sy:
            self.pixels[x][y] = on_or_name

    def apply_block_brush(self, gx, gy, on_or_name):
        bx = (gx // 2) * 2
        by = (gy // 2) * 2
        changed = []
        for dx in range(2):
            for dy in range(2):
                x = bx + dx
                y = by + dy
                if 0 <= x < self.sx and 0 <= y < self.sy:
                    self.set_pixel(x, y, on_or_name)
                    changed.append((x, y))
        self.update_image(full=False, changed_coords=changed)

    def export_png(self, filename):
        self.image.save(filename, "PNG")

    def export_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.pixels, f)

    def load_json(self, filename):
        with open(filename, "r") as f:
            self.pixels = json.load(f)
        self.update_image(full=True)

    def from_image(self, path, method="combined"):
        img_gray = Image.open(path).convert("L").resize((self.sx, self.sy), Image.Resampling.LANCZOS)

        if method == "floyd":
            dithered = floyd_steinberg_dither(img_gray).convert("L")
            arr = np.array(dithered, dtype=np.uint8)
            for y in range(self.sy):
                for x in range(self.sx):
                    self.pixels[x][y] = arr[y, x] > 128

        elif method == "sauvola":
            bin_img = sauvola_threshold(img_gray, window=9, k=0.3)
            arr = np.array(bin_img, dtype=np.uint8)
            for y in range(self.sy):
                for x in range(self.sx):
                    self.pixels[x][y] = arr[y, x] > 128

        else:
            bin_img = sauvola_threshold(img_gray, window=9, k=0.3)
            dither_img = smooth_dither(img_gray)
            bin_arr = np.array(bin_img, dtype=np.float32) / 255.0
            dith_arr = np.array(dither_img, dtype=np.float32) / 255.0
            combined = (bin_arr * dith_arr > 0.5).astype(np.uint8) * 255
            for y in range(self.sy):
                for x in range(self.sx):
                    self.pixels[x][y] = combined[y, x] > 128

        self.update_image(full=True)

    def from_image_wool(self, path):
        if not self.wool_avg_colors:
            raise RuntimeError("No wool textures loaded in ./blocks. Please add block PNGs next to the script.")

        img = Image.open(path).convert("RGB").resize((self.sx, self.sy), Image.Resampling.LANCZOS)
        arr_img = np.array(img, dtype=np.float32)

        wool_names = list(self.wool_avg_colors.keys())
        avgs = np.stack([self.wool_avg_colors[n] for n in wool_names], axis=0)

        for y in range(self.sy):
            for x in range(self.sx):
                px = arr_img[y, x]
                dists = np.linalg.norm(avgs - px, axis=1)
                idx = int(dists.argmin())
                chosen = wool_names[idx]
                self.pixels[x][y] = chosen

        self.update_image(full=True)


class LampApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Redstone Lamp Editor")
        self.grid_size = 16
        self.brush_size = 1
        self.default_wool_for_drawing = "white_wool"

        self.lamp = LampScreen(self.grid_size, self.grid_size)
        self.drawing = True
        self.erasing = False
        self.bucket_mode = False
        self.mouse_down = False

        self.label = QLabel()
        self.label.setPixmap(self.get_pixmap())
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_png.clicked.connect(self.export_png)
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_load_json = QPushButton("Load JSON")
        self.btn_load_json.clicked.connect(self.load_json)
        self.btn_img2lamp = QPushButton("Image → Lamp")
        self.btn_img2lamp.clicked.connect(self.img2lamp)
        self.btn_img2wool = QPushButton("Image → block-art")
        self.btn_img2wool.clicked.connect(self.img2wool)
        self.btn_clear_canvas = QPushButton("Clear Canvas")
        self.btn_clear_canvas.clicked.connect(self.clear_canvas)
        self.btn_bucket = QPushButton("Bucket Fill")
        self.btn_bucket.setCheckable(True)
        self.btn_bucket.clicked.connect(self.toggle_bucket)
        self.btn_export_world = QPushButton("Export World ZIP")
        self.btn_export_world.clicked.connect(self.export_to_minecraft_zip)

        self.chk_draw = QCheckBox("Drawing")
        self.chk_draw.setChecked(True)
        self.chk_draw.stateChanged.connect(self.toggle_drawing)
        self.chk_erase = QCheckBox("Erase")
        self.chk_erase.stateChanged.connect(self.toggle_erase)

        self.brush1 = QRadioButton("1×1 Brush")
        self.brush2 = QRadioButton("2×2 Brush")
        self.brush1.setChecked(True)
        self.brush_group = QButtonGroup()
        self.brush_group.addButton(self.brush1)
        self.brush_group.addButton(self.brush2)
        self.brush1.toggled.connect(lambda: self.set_brush_size(1))
        self.brush2.toggled.connect(lambda: self.set_brush_size(2))

        brush_layout = QHBoxLayout()
        brush_layout.addWidget(self.brush1)
        brush_layout.addWidget(self.brush2)
        brush_layout.addStretch()

        self.slider_label = QLabel("Grid size: 16×16")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(90)
        self.slider.setValue(8)
        self.slider.valueChanged.connect(self.change_grid_size)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(2, 180)
        self.spinbox.setSingleStep(2)
        self.spinbox.setValue(16)
        self.spinbox.valueChanged.connect(
            lambda v: self.slider.setValue(v // 2)
        )
        self.slider.valueChanged.connect(
            lambda v: self.spinbox.setValue(v * 2)
        )

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.spinbox)

        button_layout = QHBoxLayout()
        for btn in [
            self.btn_export_png, self.btn_export_json, self.btn_load_json,
            self.btn_img2lamp, self.btn_img2wool, self.btn_clear_canvas, self.btn_bucket,
            self.btn_export_world
        ]:
            button_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(slider_layout)
        layout.addLayout(brush_layout)
        layout.addWidget(self.chk_draw)
        layout.addWidget(self.chk_erase)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.update_preview()

    def set_brush_size(self, size):
        self.brush_size = size

    def change_grid_size(self, slider_value):
        value = slider_value * 2
        self.grid_size = value
        self.slider_label.setText(f"Grid size: {value}×{value}")
        self.lamp = LampScreen(value, value)
        self.update_preview()

    def mouse_press(self, event):
        if not (self.drawing or self.bucket_mode or self.erasing):
            return
        self.mouse_down = True
        self.paint_at(event)

    def mouse_move(self, event):
        if self.mouse_down and (self.drawing or self.erasing):
            self.paint_at(event)

    def mouse_release(self, event):
        self.mouse_down = False

    def paint_at(self, event):
        pos = event.position().toPoint()
        pixmap = self.label.pixmap()
        if not pixmap:
            return

        img_w = self.lamp.image.width
        scale_factor = pixmap.width() / img_w
        gx = int(pos.x() / (self.lamp.tile_size * scale_factor))
        gy = int(pos.y() / (self.lamp.tile_size * scale_factor))

        if 0 <= gx < self.lamp.sx and 0 <= gy < self.lamp.sy:
            if self.bucket_mode:
                target = self.lamp.pixels[gx][gy]
                if isinstance(target, str):
                    new_state = False if target else self.default_wool_for_drawing
                else:
                    new_state = not bool(target)
                self.bucket_fill(gx, gy, target, new_state)
            elif self.brush_size == 2:
                if isinstance(self.lamp.pixels[0][0], str):
                    val = False if self.erasing else self.default_wool_for_drawing
                else:
                    val = not self.erasing
                self.lamp.apply_block_brush(gx, gy, val)
            elif self.erasing:
                val = False
                self.lamp.set_pixel(gx, gy, val)
                self.lamp.update_image(full=False, changed_coords=[(gx, gy)])
            else:
                if isinstance(self.lamp.pixels[0][0], str):
                    val = self.default_wool_for_drawing
                else:
                    val = True
                self.lamp.set_pixel(gx, gy, val)
                self.lamp.update_image(full=False, changed_coords=[(gx, gy)])
            self.update_preview()

    def bucket_fill(self, x, y, target, new_state):
        if target == new_state:
            return
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < self.lamp.sx and 0 <= cy < self.lamp.sy:
                if self.lamp.pixels[cx][cy] == target:
                    self.lamp.pixels[cx][cy] = new_state
                    stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        self.lamp.update_image(full=True)

    def toggle_drawing(self):
        self.drawing = self.chk_draw.isChecked()
        if self.drawing:
            self.erasing = False
            self.bucket_mode = False
            self.chk_erase.setChecked(False)
            self.btn_bucket.setChecked(False)

    def toggle_erase(self):
        self.erasing = self.chk_erase.isChecked()
        if self.erasing:
            self.drawing = False
            self.bucket_mode = False
            self.chk_draw.setChecked(False)
            self.btn_bucket.setChecked(False)

    def toggle_bucket(self):
        self.bucket_mode = self.btn_bucket.isChecked()
        if self.bucket_mode:
            self.drawing = False
            self.erasing = False
            self.chk_draw.setChecked(False)
            self.chk_erase.setChecked(False)

    def clear_canvas(self):
        self.lamp.pixels = [[False for _ in range(self.lamp.sy)] for _ in range(self.lamp.sx)]
        self.lamp.update_image(full=True)
        self.update_preview()

    def get_pixmap(self):
        data = self.lamp.image.tobytes("raw", "RGBA")
        qimg = QImage(data, self.lamp.image.width, self.lamp.image.height, QImage.Format.Format_RGBA8888)
        size = 512
        return QPixmap.fromImage(qimg).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)

    def update_preview(self):
        self.lamp.update_image(full=True)
        self.label.setPixmap(self.get_pixmap())

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Files (*.png)")
        if path:
            self.lamp.export_png(path)
            QMessageBox.information(self, "Saved", f"PNG saved to:\n{path}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
        if path:
            self.lamp.export_json(path)
            QMessageBox.information(self, "Saved", f"JSON saved to:\n{path}")

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load JSON", "", "JSON Files (*.json)")
        if path:
            self.lamp.load_json(path)
            self.update_preview()

    def img2lamp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        value = self.slider.value() * 2
        self.grid_size = value
        self.slider_label.setText(f"Grid size: {value}×{value}")
        self.lamp = LampScreen(value, value)
        self.mode = "lamp"
        method = "combined"
        self.lamp.from_image(path, method=method)
        self.update_preview()
        QMessageBox.information(self, "Image Imported", f"Image is loaded and converted to {value}×{value} using '{method}' (lamp mode).")


    def img2wool(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        value = self.slider.value() * 2
        self.grid_size = value
        self.slider_label.setText(f"Grid size: {value}×{value}")
        self.lamp = LampScreen(value, value)
        self.mode = "wool"
        try:
            self.lamp.from_image_wool(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to convert image to wool: {e}")
            return
        self.update_preview()
        QMessageBox.information(self, "Image Imported", f"Image is converted to {value}×{value} (wool mapping stored).")


    def export_to_minecraft_zip(self):
        import nbtlib
        from nbtlib import load, Int, String
        zip_path, _ = QFileDialog.getSaveFileName(self, "Save Minecraft World ZIP", "", "Zip Files (*.zip)")
        if not zip_path:
            return

        script_dir = Path(__file__).parent
        base_world_dir = script_dir / "base_world"
        if not base_world_dir.exists():
            QMessageBox.warning(self, "Error", f"Base world folder '{base_world_dir}' does not exist!")
            return

        new_world_name = Path(zip_path).stem
        world_copy = Path(tempfile.mkdtemp()) / new_world_name
        shutil.copytree(base_world_dir, world_copy)

        level_dat_path = world_copy / "level.dat"
        if level_dat_path.exists():
            level_dat = load(level_dat_path)
            level_dat["Data"]["LevelName"] = String(new_world_name)
            level_dat["Data"]["SpawnX"] = Int(0)
            level_dat["Data"]["SpawnY"] = Int(-55)
            level_dat["Data"]["SpawnZ"] = Int(0)
            level_dat.save()

        self.lamp.image.save(world_copy / "icon.png", "PNG")

        dp = world_copy / "datapacks" / "lamp_pack"
        (dp / "data" / "lamp_pack" / "functions").mkdir(parents=True, exist_ok=True)
        with open(dp / "pack.mcmeta", "w") as f:
            json.dump({"pack": {"pack_format": 8, "description": "Lamp Art / Wool Art"}}, f)

        w, h = self.lamp.sx, self.lamp.sy
        base_y = -8
        lamp_y = -7
        start_x = -(w // 2)
        start_z = -(h // 2)
        cmds = []

        for y in range(h):
            for x in range(w):
                wx = start_x + x
                wz = start_z + y
                val = self.lamp.pixels[x][y]
                if self.mode == "lamp":
                    if bool(val):
                        cmds.append(f"setblock {wx} {base_y} {wz} minecraft:redstone_block")
                    else:
                        cmds.append(f"setblock {wx} {base_y} {wz} minecraft:air")
                    cmds.append(f"setblock {wx} {lamp_y} {wz} minecraft:redstone_lamp")
                else:
                    if isinstance(val, str) and val:
                        mc_block = f"minecraft:{val}"
                        cmds.append(f"setblock {wx} {base_y} {wz} minecraft:air")
                        cmds.append(f"setblock {wx} {lamp_y} {wz} {mc_block}")
                    else:
                        cmds.append(f"setblock {wx} {base_y} {wz} minecraft:air")
                        cmds.append(f"setblock {wx} {lamp_y} {wz} minecraft:air")

        sx = start_x + w // 2
        sy = -58
        sz = start_z + h // 2
        cmds.append(f"setworldspawn {sx} {sy} {sz}")
        cmds.append(f"tp @a {sx} {sy} {sz} facing {sx} {lamp_y} {start_z + h//2}")

        with open(dp / "data" / "lamp_pack" / "functions" / "load.mcfunction", "w") as f:
            f.write("\n".join(cmds))

        tag = dp / "data" / "minecraft" / "tags" / "functions"
        tag.mkdir(parents=True, exist_ok=True)
        with open(tag / "load.json", "w") as f:
            json.dump({"values": ["lamp_pack:load"]}, f)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(world_copy):
                for file in files:
                    full = Path(root) / file
                    z.write(full, full.relative_to(world_copy.parent))

        shutil.rmtree(world_copy.parent)
        QMessageBox.information(self, "Export done", f"World ZIP saved to:\n{zip_path}")




def main():
    app = QApplication(sys.argv)
    window = LampApp()
    window.resize(600, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
