"""tools/make_preview_images.py —— 生成输入图片预览（纯标准库，无需 paddle）

这只是给老师"先看一眼长什么样"的占位图：用 5x7 点阵字体画出 0-9。
真正训练后，在 AI Studio 运行推理会生成基于真实 MNIST 样本的示例图。

运行：python tools/make_preview_images.py
"""


import os
import struct
import zlib

# 5 列 x 7 行点阵字体（1=笔画，0=空白）
FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def write_png(path, width, height, pixels):
    """写入一张 8 位灰度 PNG（pixels 为长度 width*height 的字节串，0-255）。"""
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width] for y in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def render_digit(digit, size=28):
    """把点阵放大到 size×size 的灰度图（白字黑底，贴合 MNIST 风格）。"""
    grid = FONT[str(digit)]
    rows, cols = len(grid), len(grid[0])
    img = bytearray([0]) * (size * size)
    cell = size // max(rows, cols)
    off_x = (size - cols * cell) // 2
    off_y = (size - rows * cell) // 2
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch == "1":
                for dy in range(cell):
                    for dx in range(cell):
                        yy = off_y + r * cell + dy
                        xx = off_x + c * cell + dx
                        if 0 <= yy < size and 0 <= xx < size:
                            img[yy * size + xx] = 255
    return bytes(img)


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "outputs", "examples", "preview")
    os.makedirs(base, exist_ok=True)
    for d in range(10):
        write_png(os.path.join(base, f"digit_{d}.png"), 28, 28, render_digit(d))

    # 拼图：10 个数字横排
    cell, gap = 28, 4
    W = 10 * cell + 9 * gap
    montage = bytearray([0]) * (W * cell)
    for d in range(10):
        dig = render_digit(d)
        ox = d * (cell + gap)
        for yy in range(cell):
            for xx in range(cell):
                montage[yy * W + (ox + xx)] = dig[yy * cell + xx]
    write_png(os.path.join(base, "all_digits.png"), W, cell, bytes(montage))

    with open(os.path.join(base, "README.txt"), "w", encoding="utf-8") as f:
        f.write("本目录为'输入图片示例'预览（纯标准库生成的占位图，非真实 MNIST）。\n")
        f.write("在百度 AI Studio 运行 `python main.py --mode inference --run_name <你的run名>`\n")
        f.write("后，会在 outputs/examples/ 下生成基于真实 MNIST 测试样本的 sample_*.png 与 predictions.png。\n")
    print(f"预览图已生成：{os.path.abspath(base)}")


if __name__ == "__main__":
    main()
