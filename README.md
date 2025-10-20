# Redstone Lamp Editor

A **PyQt6-based pixel editor** for making Minecraft-art.

---

## Usage

Run the application with:

```bash
python main.py
```

*(or whatever filename you saved the script as)*

---

## Controls

| Action                          | Description                       |
| ------------------------------- | --------------------------------- |
| Click & drag (Drawing mode) | Turn lamps **on**                                     |
| Click & drag (Erase mode)    | Turn lamps **off**                                   |
| Drawing mode                 | Toggle drawing on/off                                |
| Erase mode                   | Toggle erasing on/off                                |
| Export PNG                   | Save the current grid as a PNG                       |
| Export JSON                  | Save current lamp state as JSON                      |
| Load JSON                    | Load a saved lamp state                              |
| Image → Lamp                 | Convert an image into a lamp grid                    |
| Bucket                       | Bucket tool                                          |
| Clear canvas                 | Clear canvas tool                                    |
| Image → Blocks               | Convert an image into a pixelart of minecraft blocks |

---

## Project Structure

```
redstone-lamp-editor/
│
├── lamp_editor.py
├── requirements.txt
├── Blocks/
  └── Red_wool
  └── Etc...
├── Redstone_lamp.png
├── Redstone_lamp_on.png
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

## Examples

![guy1](examples/guy1.png)
![guy2](examples/guy2.png)
![guy3](examples/guy3.png)
![m1](examples/m1.png)
![m2](examples/m2.png)
![shrek1](examples/shrek1.png)
![shrek2](examples/shrek2.png)
![shrek3](examples/shrek3.png)
