"""custom_input.py —— 把「任意尺寸 PNG」预处理成模型能吃的 MNIST 风格输入

【为什么要有这个模块】模型只认 MNIST 那种图：28×28、**黑底白字**、笔迹占约 20×20、
并按灰度质心居中。直接拿白底黑字的图（纸上手写拍照、画图软件导出）去喂，
准确率会莫名其妙很低——问题出在预处理，不在模型。本模块补齐这段"最后一公里"。

【零 paddle 依赖】只用 Pillow + numpy，所以能脱离深度学习环境单独自检：
    python custom_input.py inputs          # 逐张打印预处理诊断，不写任何文件

处理链（与 MNIST 的制作方式对齐）：
  ① 灰度化（大图等比降采样，避免卡顿）
  ② 极性判定：边缘中位数偏亮 → 判定为白底黑字 → 反色成黑底白字
  ③ Otsu 自动阈值分离笔迹（不靠手调阈值，适配不同扫描/拍照对比度）
  ④ 裁出笔迹外接框  ⑤ 等比缩放到长边 20px
  ⑥ 按灰度质心平移到 28×28 画布中心（MNIST 就是质心居中，比对齐外框更稳）
  ⑦ /255 后按 mean=0.1307, std=0.3081 标准化，形状 [1,28,28] float32
"""


import os

import numpy as np

# 与 data_utils.get_transforms() 的 Normalize 参数保持一致（那里 import paddle，故不复用）
MEAN, STD = 0.1307, 0.3081
TARGET_LONG_SIDE = 20          # MNIST 数字笔迹约占 20×20 外接框
CANVAS = 28                    # 模型输入边长
SUPPORTED_EXTS = (".png",)     # 课程要求统一用 PNG，其他图片格式会被忽略并提示
NEAR_MISS_EXTS = (".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")
_MAX_READ_SIDE = 1200          # 手机原图先等比缩到最长边 1200，再做阈值/质心运算


# ---------------------------------------------------------------- 读图与统计

def _read_gray(path):
    """读成 float32 灰度矩阵 [H,W]（值域 0-255），并带上原始尺寸信息。"""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("读取用户图片需要 Pillow：pip install Pillow") from e

    with Image.open(path) as im:
        im.load()                                              # 尽早暴露损坏文件
        fmt, (w, h) = im.format, im.size
        if max(w, h) > _MAX_READ_SIDE:
            r = _MAX_READ_SIDE / float(max(w, h))
            im = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
        arr = np.asarray(im.convert("L"), dtype=np.float32)     # 带色彩/透明通道一律转灰度
    return arr, {"format": fmt or "?", "orig_size": [int(w), int(h)],
                 "work_size": [int(arr.shape[1]), int(arr.shape[0])]}


def _background_level(a):
    """取四周一圈的中位数当作背景亮度（数字写在中间，边框基本是纸/背景）。"""
    k = max(1, min(a.shape) // 20)
    border = np.concatenate([a[:k].ravel(), a[-k:].ravel(), a[:, :k].ravel(), a[:, -k:].ravel()])
    return float(np.median(border))


def _otsu_threshold(a):
    """Otsu 大津法：遍历 256 个候选阈值，取类间方差最大的那个，自动适配对比度。"""
    hist, _ = np.histogram(a, bins=256, range=(0.0, 256.0))
    p = hist.astype(np.float64) / float(max(1, hist.sum()))
    idx = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)                       # 累计概率
    mu = np.cumsum(p * idx)                    # 累计均值
    den = omega * (1.0 - omega)
    sigma_b2 = np.where(den > 0, (mu[-1] * omega - mu) ** 2 / np.where(den > 0, den, 1.0), 0.0)
    return int(np.argmax(sigma_b2))


# ---------------------------------------------------------------- 几何变换

def _resize_long_side(arr, long_side):
    """等比缩放，使长边 = long_side（保持笔画粗细比例，别把 7 拉成 1）。"""
    from PIL import Image
    h, w = arr.shape
    s = long_side / float(max(h, w))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).resize((nw, nh), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def _center_on_canvas(arr):
    """按灰度质心把笔迹平移到 28×28 黑画布正中；越界部分裁掉（笔迹已被限制在 20px，一般不会）。"""
    h, w = arr.shape
    tot = float(arr.sum())
    if tot > 0:
        yy, xx = np.indices(arr.shape)
        cy = float((arr * yy).sum()) / tot
        cx = float((arr * xx).sum()) / tot
    else:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    oy, ox = int(round(CANVAS / 2.0 - cy)), int(round(CANVAS / 2.0 - cx))
    canvas = np.zeros((CANVAS, CANVAS), dtype="uint8")
    ys0, ys1 = max(0, -oy), min(h, CANVAS - oy)
    xs0, xs1 = max(0, -ox), min(w, CANVAS - ox)
    if ys1 > ys0 and xs1 > xs0:
        canvas[oy + ys0:oy + ys1, ox + xs0:ox + xs1] = np.clip(arr[ys0:ys1, xs0:xs1], 0, 255).astype("uint8")
    return canvas


# ---------------------------------------------------------------- 对外接口

def preprocess_png(path):
    """一张用户 PNG → dict(x=[1,28,28] float32 已标准化, gray=uint8[28,28] 送模型前的样子, meta=诊断)。

    图里没笔迹 / 背景分不出来时抛 ValueError，消息即"该改哪里"，由上层逐张跳过。
    """
    a, meta = _read_gray(path)

    # 先查对比度：整张图灰度几乎没有起伏，说明是空白页/纯黑图/没对上焦，后面的阈值就无从下手
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 32:
        raise ValueError("整张图几乎没有对比度（灰度范围 %.0f~%.0f）：可能是空白图或背景与笔迹同色，"
                         "请确认图里有一个颜色分明的数字" % (lo, hi))

    bg = _background_level(a)
    inverted = bg > 127                                   # 白底黑字 → 反色成 MNIST 的黑底白字
    if inverted:
        a = 255.0 - a

    thr = _otsu_threshold(a)
    mask = a > thr
    ink = float(mask.mean())
    if ink < 1e-4:
        raise ValueError("没检测到笔迹（Otsu 阈值 %d，前景占比 %.5f）：确认图里写了一个数字，"
                         "或把笔画画得更粗更黑" % (thr, ink))
    if ink > 0.9:
        raise ValueError("整张图几乎都判成了笔迹（前景占比 %.2f）：背景没分离出来，"
                         "请改成单字、白底黑字或黑底白字，去掉阴影与背景杂物" % ink)

    ys, xs = np.where(mask)
    r0, c0, r1, c1 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
    small = _resize_long_side(a[r0:r1, c0:c1], TARGET_LONG_SIDE)
    canvas = _center_on_canvas(small)

    x = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    meta.update({"inverted": bool(inverted), "background_level": round(bg, 1),
                 "otsu_threshold": thr, "ink_ratio": round(ink, 4),
                 "bbox_xywh": [c0, r0, c1 - c0, r1 - r0],
                 "resized_to": [int(small.shape[1]), int(small.shape[0])],
                 "canvas": [CANVAS, CANVAS]})
    return {"x": x[None], "gray": canvas, "meta": meta}


def iter_user_images(directory):
    """返回 (可识别的 PNG 路径列表, 看起来是图片但格式不对的文件名)，按文件名排序保证可复现。

    第二个列表只报图片类后缀（jpg/webp…），README.txt 之类的文本不报警，避免噪声。
    """
    if not os.path.isdir(directory):
        return [], []
    files, ignored = [], []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            files.append(path)
        elif ext in NEAR_MISS_EXTS:
            ignored.append(name)
    return files, ignored


def label_from_filename(path):
    """文件名以数字开头（`7.png` / `3_我写的.png`）时，把该数字当作用户给的标准答案，用于算准确率。"""
    stem = os.path.splitext(os.path.basename(path))[0].strip()
    return int(stem[0]) if stem[:1].isdigit() else None


# ---------------------------------------------------------------- 命令行自检

if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "inputs"
    pics, skipped = iter_user_images(target)
    print("扫描目录：%s" % os.path.abspath(target))
    if not pics:
        print("  没有 PNG 可测。把你的手写数字图放进这个目录（一张图一个数字）再重试。")
    for name in skipped:
        print("  [忽略] %s —— 仅支持 %s，请转成 PNG 再放进来" % (name, "/".join(SUPPORTED_EXTS)))
    for path in pics:
        try:
            r = preprocess_png(path)
        except Exception as e:
            print("  [跳过] %-20s %s" % (os.path.basename(path), e))
            continue
        m = r["meta"]
        print("  [OK] %-20s 原尺寸%s 反色=%s Otsu=%-3d 前景=%.3f 外接框=%s → 缩放%s 峰值%d"
              % (os.path.basename(path), m["orig_size"], "是" if m["inverted"] else "否",
                 m["otsu_threshold"], m["ink_ratio"], m["bbox_xywh"], m["resized_to"], int(r["gray"].max())))
    print("说明：[OK] 的图都能送进模型；上面这些诊断也会写进 outputs/custom/custom_results.json。")
