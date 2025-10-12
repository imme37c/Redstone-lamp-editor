# Redstone Lamp Editor

A simple **PyQt6-based pixel editor** for making Minecraft-Redstone Lamp patterns.

---

## Usage

Run the application with:

```bash
python lamp_editor.py
```

*(or whatever filename you saved the script as)*

---

## Controls

| Action                          | Description                       |
| ------------------------------- | --------------------------------- |
| Click & drag (Drawing mode) | Turn lamps **on**                 |
| Click & drag (Erase mode)    | Turn lamps **off**                |
| Drawing mode                 | Toggle drawing on/off             |
| Erase mode                   | Toggle erasing on/off             |
| Export PNG                   | Save the current grid as a PNG    |
| Export JSON                  | Save current lamp state as JSON   |
| Load JSON                    | Load a saved lamp state           |
| Image → Lamp                | Convert an image into a lamp grid |
| Bucket tool | Bucket tool|
| Clear canvas | clear canvas tool |

---

## Project Structure

```
redstone-lamp-editor/
│
├── lamp_editor.py
├── requirements.txt
└── README.md
```

---

## Platforms Supported

| Platform   | Supported |
| ---------- | --------- |
| Windows | ✅         |
| macOS   | ✅         |
| Linux   | ✅         |

---

## License

This project is free to use and modify for personal or educational purposes.
Credits are appreciated if you share or modify the project.

---

## Credits

This project was inspired by [mattbatwings](https://github.com/mattbatwings),
who created [lampsim](https://github.com/mattbatwings/lampsim) — a Redstone lamp simulator for Minecraft.
