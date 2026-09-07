"""data_ops.py —— 数据运维巡检（资产台账 / 一致性校验 / 告警 / 可视化看板）

本模块把 README §1.1 那张「产物保存位置总览」表变成可自动执行的体检：
扫描 logs/、data/、outputs/ 三个区域的**真实产物**，从 5 个运维维度打分，
输出 JSON + Markdown 报告、matplotlib 看板 PNG、零依赖 HTML 看板。

【只读原则】只扫描、只校验、只出报告，**绝不修改或删除任何训练产物**。
清理/归档动作仅以「建议命令」形式写进报告，由人确认后自己执行。

【零重依赖】巡检本体只用 Python 标准库（gzip/hashlib/json/re），
matplotlib 缺失时自动跳过 PNG，Markdown / HTML 报告照常生成，
所以任何同学的本机环境都能跑出成果、也方便教师离线批改。

五个运维维度（加权合成健康度总分）：
  资产完整性 completeness  —— run 目录 7 类产物是否齐全
  一致性校验 consistency   —— 日志 / history / config / 数据留档 能否互相印证
  可追溯性 traceability    —— 参数快照、数据指纹、时间戳是否支撑复现
  数据源健康 source_health  —— data/ 的 MNIST .gz 是否完整合法、有无基线漂移
  存储治理 storage_gov     —— 体积、0 字节、重复日志、超期 run、孤儿文件

用法（推荐走统一入口）：
    python main.py --mode ops
    python main.py --mode ops --retention_days 7 --run_name demo
"""


import glob
import gzip
import hashlib
import html as _html
import json
import os
import re
import struct
import time

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
LEVEL_RANK = {ERROR: 0, WARN: 1, INFO: 2}
LEVEL_COLOR = {ERROR: "#e5484d", WARN: "#f5a623", INFO: "#3b82f6"}

# (维度 key, 中文名, 权重) —— 权重之和须为 1
DIMENSIONS = [
    ("completeness", "资产完整性", 0.25),
    ("consistency", "一致性校验", 0.25),
    ("traceability", "可追溯性", 0.20),
    ("source_health", "数据源健康", 0.15),
    ("storage_gov", "存储治理", 0.15),
]
PENALTY = {ERROR: 35, WARN: 12, INFO: 3}

# run 目录里应当存在的产物（对应 train.py 的保存逻辑）
RUN_ARTIFACTS = [
    {"key": "model", "file": "model.pdparams", "label": "最终模型权重", "level": ERROR, "dim": "completeness"},
    {"key": "best_model", "file": "best_model.pdparams", "label": "最佳模型权重", "level": ERROR, "dim": "completeness"},
    {"key": "config", "file": "config.json", "label": "参数快照", "level": ERROR, "dim": "completeness"},
    {"key": "history", "file": "history.json", "label": "训练过程数据", "level": WARN, "dim": "completeness"},
    {"key": "curves", "file": "curves.png", "label": "训练曲线图", "level": WARN, "dim": "completeness"},
    {"key": "record", "file": "dataset_record.json", "label": "数据集留档", "level": WARN, "dim": "traceability"},
]
KNOWN_FILES = {a["file"] for a in RUN_ARTIFACTS} | {"inference_results.json"}

# MNIST 四个原始文件；idx_magic/维度用于解析校验（2049=idx1标签, 2051=idx3图像）
MNIST_FILES = [
    ("train-images-idx3-ubyte.gz", "train", 2051, 60000),
    ("train-labels-idx1-ubyte.gz", "train", 2049, 60000),
    ("t10k-images-idx3-ubyte.gz", "test", 2051, 10000),
    ("t10k-labels-idx1-ubyte.gz", "test", 2049, 10000),
]
PADDLE_FALLBACK = os.path.expanduser("~/.cache/paddle/dataset/mnist")

LOG_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*train_loss=([\d.]+)\s+train_acc=([\d.]+)"
    r"\s*\|\s*val_loss=([\d.]+)\s+val_acc=([\d.]+)"
)


# ---------------------------------------------------------------- 基础工具

def human_bytes(n):
    """把字节数格式化成人类可读体积。"""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _file_info(path):
    st = os.stat(path)
    return {
        "size": st.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "mtime_ts": st.st_mtime,
        "abs": os.path.abspath(path),
    }


def _load_json(path):
    """读 json；文件为空/损坏时返回 (None, 错误说明)，交给上层报 ERROR。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败：{e}"
    except OSError as e:
        return None, f"读取失败：{e}"


def parse_idx_gz(path):
    """解析 MNIST 的 .gz（IDX 格式）头部，并流式统计解压字节数。

    只读头部 + 累加长度，不把整份图像载入内存，9.9MB 的训练集也很快。
    返回 {"ok", "magic", "count", "rows", "cols", "declared", "actual"}。
    """
    out = {"ok": False, "magic": None, "count": None, "rows": None, "cols": None,
           "declared": None, "actual": 0, "error": None}
    try:
        with gzip.open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                out["error"] = "文件过小，非合法 IDX 流"
                return out
            magic, d0 = struct.unpack(">II", head)
            ndim = {2049: 1, 2051: 3}.get(magic)
            if ndim is None:
                out["error"] = f"IDX magic 非法：{magic}（应为 2049 标签 / 2051 图像）"
                return out
            out["magic"], out["count"] = magic, d0
            if ndim > 1:
                extra = f.read((ndim - 1) * 4)
                if len(extra) < (ndim - 1) * 4:
                    out["error"] = "IDX 头部被截断"
                    return out
                dims = [d0] + list(struct.unpack(f">{ndim - 1}I", extra))
                out["rows"], out["cols"] = dims[1], dims[2]
                declared = 1
                for d in dims:
                    declared *= d
            else:
                declared = d0
            out["declared"] = declared
            actual = 0
            for block in iter(lambda: f.read(1 << 20), b""):
                actual += len(block)
            out["actual"] = actual
            out["ok"] = (actual == declared)
            if not out["ok"]:
                out["error"] = f"解压长度与头部声明不符：实际 {actual} != 声明 {declared}（文件被截断）"
            return out
    except (OSError, EOFError, struct.error) as e:
        out["error"] = f"读取/解压失败：{e}"
        return out


def _issue(code, dim, level, target, message, suggestion=""):
    return {"code": code, "dim": dim, "level": level, "target": target,
            "message": message, "suggestion": suggestion}


# ---------------------------------------------------------------- 数据源体检

def audit_dataset_source(cfg, issues, logger):
    """体检 MNIST 原始数据：存在性 → IDX 合法性 → 与指纹基线比对是否漂移。

    指纹基线 = 首次巡检时把 md5/size/IDX 头记到 <ops_dir>/data_fingerprint.json，
    之后每次巡检比对，可发现"数据被误覆盖/下载不全/被人改动"这类静默损坏。
    """
    roots = []
    for cand in (cfg.data_dir, PADDLE_FALLBACK):
        ab = os.path.abspath(os.path.expanduser(cand))
        if ab not in roots:
            roots.append(ab)

    found_root = next((r for r in roots if os.path.isdir(r)
                       and any(os.path.exists(os.path.join(r, f[0])) for f in MNIST_FILES)), None)
    files, baseline_path = [], os.path.join(os.path.abspath(cfg.ops_dir), "data_fingerprint.json")
    baseline = {}
    if os.path.exists(baseline_path):
        baseline, err = _load_json(baseline_path)
        if err:
            issues.append(_issue("S0", "source_health", ERROR, baseline_path,
                                 f"数据指纹基线损坏：{err}", "删除该文件后用 --refresh_baseline 重建基线"))
        baseline = baseline or {}
    base_files = baseline.get("files", {})

    for fname, split, magic, expect_count in MNIST_FILES:
        path = os.path.join(found_root or roots[0], fname)
        rec = {"name": fname, "split": split, "path": os.path.abspath(path), "exists": False}
        if not os.path.exists(path):
            rec["error"] = "文件不存在"
            issues.append(_issue("S1", "source_health", ERROR, fname,
                                 f"数据集文件缺失：{rec['path']}",
                                 "运行一次训练触发自动下载，或手工补齐 MNIST 四个 .gz"))
            files.append(rec)
            continue
        rec["exists"] = True
        rec.update(_file_info(path))
        if rec["size"] == 0:
            issues.append(_issue("S2", "source_health", ERROR, fname, "数据集文件为 0 字节", "删除后重新下载"))
        idx = parse_idx_gz(path)
        rec["idx"] = {k: idx[k] for k in ("magic", "count", "rows", "cols", "declared", "actual", "ok")}
        if not idx["ok"]:
            issues.append(_issue("S3", "source_health", ERROR, fname,
                                 idx["error"] or "IDX 结构校验未通过", "该文件已损坏，删除后重新下载"))
        else:
            if idx["magic"] != magic:
                issues.append(_issue("S4", "source_health", ERROR, fname,
                                     f"IDX 类型不符：magic={idx['magic']} 应为 {magic}", "文件名与内容不匹配，可能被误替换"))
            if idx["count"] != expect_count:
                issues.append(_issue("S5", "source_health", WARN, fname,
                                     f"样本数异常：{idx['count']} != 标准 {expect_count}", "确认是否换过数据集版本，并同步更新留档记录"))
            if idx["magic"] == 2051 and (idx["rows"], idx["cols"]) != (28, 28):
                issues.append(_issue("S6", "source_health", WARN, fname,
                                     f"图像尺寸异常：{idx['rows']}x{idx['cols']} != 28x28", "与 config/模型输入层核对"))
            rec["md5"] = _md5(path)
            b = base_files.get(fname)
            if b:
                rec["baseline_md5"] = b.get("md5")
                rec["drift"] = (b.get("md5") != rec["md5"])
                if rec["drift"]:
                    issues.append(_issue("S7", "source_health", ERROR, fname,
                                         f"数据基线漂移：md5 {b.get('md5')[:8]}… → {rec['md5'][:8]}…",
                                         "确认数据变更原因；确认可信后 --refresh_baseline 重建基线"))
        files.append(rec)

    label_map = {f[0]: f for f in MNIST_FILES}
    for split in ("train", "test"):
        imgs = next((f for f in files if f["split"] == split and f.get("idx", {}).get("magic") == 2051), None)
        labs = next((f for f in files if f["split"] == split and f.get("idx", {}).get("magic") == 2049), None)
        if imgs and labs and imgs["exists"] and labs["exists"]:
            ic, lc = imgs["idx"]["count"], labs["idx"]["count"]
            if ic is not None and ic != lc:
                issues.append(_issue("S8", "source_health", ERROR, f"{split} split",
                                     f"图像/标签样本数不一致：images={ic} labels={lc}", "标签与图像必须一一对应"))

    if found_root is None:
        issues.append(_issue("S9", "source_health", WARN, cfg.data_dir,
                             f"未找到任何 MNIST 数据目录（已查 {', '.join(roots)}）",
                             "先跑一次 python main.py --mode train --run_name demo 生成数据"))
    elif os.path.abspath(found_root) != os.path.abspath(cfg.data_dir):
        issues.append(_issue("S10", "source_health", WARN, found_root,
                             f"实际数据目录为 {found_root}，与 config.data_dir={cfg.data_dir} 不一致"
                             f"（说明当时触发了退回 Paddle 默认缓存的降级分支）",
                             "在报告中注明降级；如需固定路径请补齐 config.data_dir 下的文件"))

    # 基线写入时机：首次巡检自动建立；或基线里缺文件时补齐。
    # 注：发现漂移时**不自动覆盖基线**（否则就抹掉了证据），
    # 需人工确认后用 --refresh_baseline 显式重建。
    if found_root:
        needs_write = cfg.refresh_baseline or not base_files or \
            any(f.get("md5") and f["name"] not in base_files for f in files)
        if needs_write and any(f.get("md5") for f in files):
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            payload = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "root": found_root,
                "files": {f["name"]: {"md5": f.get("md5"), "size": f.get("size"),
                                      "count": f.get("idx", {}).get("count")}
                          for f in files if f.get("md5")},
            }
            with open(baseline_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            if not base_files:
                logger.info("已建立数据指纹基线：%s（%d 个文件）", baseline_path, len(payload["files"]))
                issues.append(_issue("S11", "source_health", INFO, baseline_path,
                                     "首次巡检：已建立数据指纹基线，后续可检测数据漂移", ""))
            elif cfg.refresh_baseline:
                logger.info("已按 --refresh_baseline 重建数据指纹基线")

    total = sum(f.get("size", 0) for f in files)
    return {"root": found_root, "searched": roots, "files": files,
            "total_bytes": total, "fingerprint_path": baseline_path,
            "ok_count": sum(1 for f in files if f["exists"] and f.get("idx", {}).get("ok"))}


# ---------------------------------------------------------------- run 台账

def _parse_train_log(path):
    """解析 train_*.log，返回逐轮指标列表（用于与 history.json 交叉核对）。"""
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LOG_EPOCH_RE.search(line)
                if m:
                    rows.append({"epoch": int(m.group(1)), "total": int(m.group(2)),
                                 "train_loss": float(m.group(3)), "train_acc": float(m.group(4)),
                                 "val_loss": float(m.group(5)), "val_acc": float(m.group(6))})
    except OSError:
        return None
    return rows


def scan_run(run_dir, source, issues, cfg):
    """扫描一个 run 目录，产出台账 + 该 run 的一致性/可追溯性/存储治理告警。"""
    name = os.path.basename(os.path.normpath(run_dir))
    run = {"name": name, "path": os.path.abspath(run_dir), "files": {}, "artifacts": {},
           "logs": [], "history": None, "config": None, "record": None,
           "log_rows": None, "final": {}, "dims": {}}

    for entry in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if os.path.isdir(entry):
            issues.append(_issue("R0", "storage_gov", WARN, f"{name}/{os.path.basename(entry)}",
                                 "run 目录内出现子目录，不符合扁平产物布局", "确认是否误建目录"))
            continue
        run["files"][os.path.basename(entry)] = _file_info(entry)

    # ---- ① 资产完整性
    present, missing = [], []
    for art in RUN_ARTIFACTS:
        got = run["files"].get(art["file"])
        run["artifacts"][art["key"]] = bool(got)
        (present if got else missing).append(art["file"])
        if not got:
            issues.append(_issue("C1", art["dim"], art["level"], f"{name}/{art['file']}",
                                 f"缺少{art['label']}（{art['file']}）",
                                 "重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录"))
    logs = sorted(k for k in run["files"] if re.match(r"train_\d{8}_\d{6}\.log$", k))
    run["logs"] = logs
    run["artifacts"]["log"] = bool(logs)
    if not logs:
        issues.append(_issue("C2", "completeness", ERROR, name,
                             "缺少训练日志 train_YYYYMMDD_HHMMSS.log", "确认 logger.get_logger 是否被调用"))
    if not logs and not present:
        run["empty"] = True
        issues.append(_issue("R1", "storage_gov", WARN, name,
                             "空 run 目录（无任何有效产物）", "确认是否为误建；建议归档后清理"))

    # ---- ② 一致性：日志 ↔ history ↔ config
    hist, hist_err = (None, None)
    if run["files"].get("history.json"):
        hist, hist_err = _load_json(os.path.join(run_dir, "history.json"))
    if hist_err:
        issues.append(_issue("N1", "consistency", ERROR, f"{name}/history.json", hist_err, "重新训练生成"))
    run["history"] = hist

    cfgd, cfg_err = (None, None)
    if run["files"].get("config.json"):
        cfgd, cfg_err = _load_json(os.path.join(run_dir, "config.json"))
    if cfg_err:
        issues.append(_issue("N2", "consistency", ERROR, f"{name}/config.json", cfg_err, "重新训练生成"))
    run["config"] = cfgd

    log_rows = _parse_train_log(os.path.join(run_dir, logs[-1])) if logs else None
    run["log_rows"] = log_rows

    if hist:
        keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        missing_keys = [k for k in keys if k not in hist]
        if missing_keys:
            issues.append(_issue("N3", "consistency", ERROR, f"{name}/history.json",
                                 f"history 缺字段：{missing_keys}", "训练逻辑可能被改动，核对 train.py"))
        else:
            lens = {k: len(hist[k]) for k in keys}
            if len(set(lens.values())) != 1:
                issues.append(_issue("N4", "consistency", ERROR, f"{name}/history.json",
                                     f"history 各序列长度不等：{lens}", "记录被截断，视为不可信 run"))
            else:
                run["final"] = {
                    "epochs": len(hist["epoch"]),
                    "train_acc": hist["train_acc"][-1], "val_acc": hist["val_acc"][-1],
                    "val_loss": hist["val_loss"][-1],
                    "best_val_acc": max(hist["val_acc"]),
                    "best_epoch": hist["epoch"][hist["val_acc"].index(max(hist["val_acc"]))],
                    "model_type": (cfgd or {}).get("model_type", "?"),
                }
                gap = round(run["final"]["train_acc"] - run["final"]["val_acc"], 4)
                run["final"]["gap"] = gap
                if gap > 0.05:
                    issues.append(_issue("N5", "consistency", WARN, name,
                                         f"泛化间隙过大 train_acc-val_acc={gap}（>0.05），疑似过拟合",
                                         "增大 dropout / 减少 epochs / 换更大结构对比验证"))
                vl = hist["val_loss"]
                if len(vl) >= 4 and vl[-1] > min(vl) * 1.1:
                    issues.append(_issue("N6", "consistency", WARN, name,
                                         f"末轮 val_loss={vl[-1]} 比最优轮 {min(vl)} 高 >10%，训练后期退化",
                                         "改用 best_model.pdparams 做推理，或加早停"))

            if log_rows is not None:
                if len(log_rows) != len(hist["epoch"]):
                    issues.append(_issue("N7", "consistency", ERROR, name,
                                         f"日志与过程数据不一致：日志 {len(log_rows)} 轮 vs history {len(hist['epoch'])} 轮",
                                         "典型的运维事故：日志被截断或产物被手工编辑，需重跑"))
                else:
                    la = log_rows[-1]["val_acc"]
                    if abs(la - hist["val_acc"][-1]) > 1e-4:
                        issues.append(_issue("N8", "consistency", WARN, name,
                                             f"末轮 val_acc 对不上：日志 {la} vs history {hist['val_acc'][-1]}",
                                             "确认 history 是否被手工改过"))
            elif logs:
                issues.append(_issue("N9", "consistency", WARN, f"{name}/{logs[-1]}",
                                     "日志中解析不到 Epoch 指标行", "确认 train.py 的日志格式未被修改"))

        if cfgd and cfgd.get("epochs") is not None and cfgd["epochs"] != len(hist["epoch"]):
            issues.append(_issue("N10", "traceability", WARN, name,
                                 f"配置声明 epochs={cfgd['epochs']} 与实际记录 {len(hist['epoch'])} 轮不符",
                                 "训练可能中断或被外部终止，此 run 不建议作为提交物"))

    if cfgd and cfgd.get("run_name") and cfgd["run_name"] != name:
        issues.append(_issue("N11", "consistency", ERROR, name,
                             f"config.run_name={cfgd['run_name']} 与目录名 {name} 不符",
                             "产物被跨 run 复制过，追溯链已断"))

    rec, rec_err = (None, None)
    if run["files"].get("dataset_record.json"):
        rec, rec_err = _load_json(os.path.join(run_dir, "dataset_record.json"))
    if rec_err:
        issues.append(_issue("N12", "traceability", ERROR, f"{name}/dataset_record.json", rec_err,
                             "重跑训练生成数据留档"))
    run["record"] = rec

    if rec:
        if source["root"] and rec.get("cache_dir"):
            declared = os.path.abspath(os.path.expanduser(rec["cache_dir"]))
            if declared != source["root"]:
                issues.append(_issue("N13", "traceability", WARN, f"{name}/dataset_record.json",
                                     f"留档声明数据目录 {rec['cache_dir']}，实际命中 {source['root']}",
                                     "留档与事实不符，追溯时以实际为准"))
        for split, key in (("train", "train_samples"), ("test", "test_samples")):
            actual = next((f["idx"]["count"] for f in source["files"]
                           if f["split"] == split and f.get("idx")), None)
            if actual is not None and rec.get(key) is not None and rec[key] != actual:
                issues.append(_issue("N14", "consistency", ERROR, f"{name}/dataset_record.json",
                                     f"台账样本数 {key}={rec[key]} 与数据实测 {actual} 不符",
                                     "台账造假/数据被替换，二者必居其一，必须重查"))
    elif not rec_err:
        issues.append(_issue("T1", "traceability", WARN, name,
                             "无 dataset_record.json，无法追溯本次用了什么数据", "重跑训练以生成数据留档"))

    # ---- ③ 可追溯性：参数快照够不够复现
    if cfgd:
        need = ["model_type", "learning_rate", "epochs", "batch_size", "optimizer", "seed"]
        absent = [k for k in need if cfgd.get(k) is None]
        if absent:
            issues.append(_issue("T2", "traceability", WARN, f"{name}/config.json",
                                 f"参数快照缺复现必需字段：{absent}", "补全 Config 字段后重跑"))
    elif not run["files"].get("config.json"):
        pass  # 缺件已在 C1 报过，避免重复计数
    if not run["artifacts"]["best_model"]:
        issues.append(_issue("T3", "traceability", INFO, name,
                             "无 best_model.pdparams，inference 模式将回退用末轮权重",
                             "推理结果可能与训练报告里的 val_acc 不一致"))

    # ---- ④ 存储治理
    for fname, info in run["files"].items():
        if info["size"] == 0:
            issues.append(_issue("G1", "storage_gov", WARN, f"{name}/{fname}", "0 字节文件（写入中断嫌疑）",
                                 "删除后重跑该 run"))
        elif fname not in KNOWN_FILES and not fname.startswith("train_"):
            issues.append(_issue("G2", "storage_gov", INFO, f"{name}/{fname}",
                                 "非预期文件（孤儿产物）", "确认是否需要保留或纳入台账表"))
    if len(logs) > 1:
        issues.append(_issue("G3", "storage_gov", INFO, name,
                             f"同一 run 有 {len(logs)} 份训练日志，说明重复写入同一 run_name",
                             "每次换 --run_name，日志才互不覆盖"))
    age_days = (time.time() - max([i["mtime_ts"] for i in run["files"].values()] or [0])) / 86400
    run["age_days"] = round(age_days, 1)
    run["total_bytes"] = sum(i["size"] for i in run["files"].values())
    if age_days > cfg.retention_days and run["total_bytes"] > 1 << 20:
        issues.append(_issue("G4", "storage_gov", WARN, name,
                             f"已 {age_days:.0f} 天未更新（保留期 {cfg.retention_days} 天）且占 "
                             f"{human_bytes(run['total_bytes'])}",
                             f"确认无需提交后可归档：tar czf {name}.tar.gz logs/{name} && rm -rf logs/{name}"))

    by_cat = {"weights": 0, "logs": 0, "images": 0, "json": 0, "other": 0}
    for fname, info in run["files"].items():
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdparams":
            cat = "weights"
        elif ext == ".log":
            cat = "logs"
        elif ext == ".png":
            cat = "images"
        elif ext == ".json":
            cat = "json"
        else:
            cat = "other"
        by_cat[cat] += info["size"]
    run["by_cat"] = by_cat
    run["human_size"] = human_bytes(run["total_bytes"])
    return run


# ------------------------------------------------- 用户自提图片入口体检（新数据接入）

def audit_custom_inputs(cfg, runs, issues):
    """体检 --mode custom 这条新数据入口：inputs/ → outputs/custom/ 是否闭环。

    学生自己写的图就是一次「新数据接入管道」，运维必须回答三个问题：
      ① 入口有多少图、多少被格式/质量挡在门外（数据质量工单）；
      ② 报告里记的输入数与目录实际图数是否一致（防止图换了、报告没重跑）；
      ③ 结果引用的模型权重还在不在（防止权重删了、报告无法复现）。
    """
    in_dir = os.path.abspath(os.path.expanduser(getattr(cfg, "image_dir", None) or "./inputs"))
    out_dir = os.path.join(os.path.abspath(cfg.output_dir), "custom")
    res_path = os.path.join(out_dir, "custom_results.json")
    info = {"image_dir": in_dir, "out_dir": out_dir, "results_path": res_path,
            "dir_exists": os.path.isdir(in_dir), "pngs": [], "wrong_ext": [],
            "exists": os.path.exists(res_path), "total_bytes": 0, "out_bytes": 0,
            "num_images": None, "succeeded": None, "failed": [], "unlabeled": None,
            "accuracy": None, "run_name": None, "model_type": None, "model_missing": None,
            "generated_at": None, "orphans": [], "by_file": {}}

    # ---- 入口侧：扫 inputs/
    if not info["dir_exists"]:
        issues.append(_issue("U1", "traceability", INFO, in_dir,
                             "自定义输入目录不存在（本次未使用 --mode custom）",
                             "把一张图一个数字的 PNG 放进该目录后运行 --mode custom"))
    else:
        pngs, ignored = [], []
        try:
            from custom_input import iter_user_images            # 需要 numpy + Pillow
            pngs, ignored = iter_user_images(in_dir)
        except Exception:
            for name in sorted(os.listdir(in_dir)):
                p = os.path.join(in_dir, name)
                if not os.path.isfile(p) or name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext == ".png":
                    pngs.append(p)
                elif ext in (".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"):
                    ignored.append(name)
        for p in pngs:
            st = os.stat(p)
            info["pngs"].append({"name": os.path.basename(p), "size": st.st_size,
                                 "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
            if st.st_size == 0:
                issues.append(_issue("U2", "source_health", ERROR, "inputs/" + os.path.basename(p),
                                     "输入 PNG 为 0 字节（导出或拷贝中断）", "重新导出该图"))
        info["wrong_ext"] = list(ignored)
        info["total_bytes"] = sum(f["size"] for f in info["pngs"])
        if ignored:
            issues.append(_issue("U3", "source_health", WARN, in_dir,
                                 "%d 个文件看着是图但格式不支持，已被排除：%s"
                                 % (len(ignored), ", ".join(ignored[:4]) + (" …" if len(ignored) > 4 else "")),
                                 "统一转成 PNG 再送进模型，否则学生容易误判为模型识别错了"))

    # ---- 成果侧：outputs/custom/
    if os.path.isdir(out_dir):
        info["out_bytes"] = sum(os.path.getsize(os.path.join(out_dir, n))
                                for n in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, n)))
    if not info["exists"]:
        if info["pngs"]:
            issues.append(_issue("U5", "completeness", WARN, res_path,
                                 "已有 %d 张输入图但没有 custom_results.json" % len(info["pngs"]),
                                 "运行：python main.py --mode custom --run_name <run名>"))
        return info

    data, err = _load_json(res_path)
    if err:
        issues.append(_issue("U6", "consistency", ERROR, res_path, "自定义推理结果损坏：" + err, "重跑 --mode custom"))
        return info

    info.update({"run_name": data.get("run_name"), "num_images": data.get("num_images"),
                 "succeeded": data.get("num_succeeded"), "accuracy": data.get("accuracy_on_labeled"),
                 "unlabeled": (data.get("num_images") or 0) - (data.get("num_labeled") or 0),
                 "generated_at": data.get("generated_at"), "model_type": data.get("model_type")})
    for it in data.get("results", []):
        name = it.get("file")
        if name:
            info["by_file"][name] = {"pred": it.get("pred"), "prob": it.get("prob"),
                                     "true": it.get("true"), "correct": it.get("correct"),
                                     "error": it.get("error")}
        if "error" in it:
            info["failed"].append({"file": it.get("file"), "error": it["error"]})
    if info["failed"]:
        issues.append(_issue("U7", "source_health", WARN, in_dir,
                             "%d 张输入图预处理失败：%s" % (len(info["failed"]),
                                                           ", ".join(str(f["file"]) for f in info["failed"][:4])
                                                           + (" …" if len(info["failed"]) > 4 else "")),
                             "按报因改图（单字、对比度、去阴影），可先用 python custom_input.py inputs 自检"))

    # ---- 交叉核对：报告 ↔ 当前输入 ↔ 模型权重
    if (info["num_images"] is not None and info["dir_exists"]
            and info["num_images"] != len(info["pngs"])):
        issues.append(_issue("U8", "consistency", WARN, res_path,
                             "报告记录的输入数 %s ≠ 目录现有图数 %d（报告生成于 %s）"
                             % (info["num_images"], len(info["pngs"]), info.get("generated_at")),
                             "输入集已变更但没重跑，该成果不能代表当前输入，请重跑 --mode custom"))
    model_path = data.get("model_path")
    if model_path and not os.path.exists(model_path):
        info["model_missing"] = model_path
        issues.append(_issue("U9", "traceability", ERROR, res_path,
                             "结果引用的模型权重已不存在：" + str(model_path),
                             "权重被清理导致成果无法复现；重跑训练后重跑推理"))
    if info["run_name"] and info["run_name"] not in [r["name"] for r in runs]:
        issues.append(_issue("U10", "traceability", WARN, res_path,
                             "结果引用的 run「%s」在 logs/ 中不存在" % info["run_name"],
                             "输入图与模型无法配对追溯；保留训练产物或重新跑"))
    if info["unlabeled"]:
        issues.append(_issue("U11", "traceability", INFO, in_dir,
                             "%d 张图文件名未以数字开头，无标准答案，不纳入准确率" % info["unlabeled"],
                             "想统计准确率就按 7.png 或 3_我写的.png 命名"))

    known = {it.get("saved_image") for it in data.get("results", []) if it.get("saved_image")}
    known |= {"custom_results.json", "predictions_custom.png"}
    if os.path.isdir(out_dir):
        info["orphans"] = [n for n in sorted(os.listdir(out_dir)) if n not in known]
        for n in info["orphans"]:
            issues.append(_issue("U12", "storage_gov", INFO, "outputs/custom/" + n,
                                 "孤儿产物（当前结果 json 里找不到对应记录）",
                                 "上次运行残留，确认后手动删除或整目录重建"))
    return info


# ---------------------------------------------------------------- 汇总与评分

def _dir_bytes(path):
    """一个目录下直接子文件的体积合计（不算入子目录，用于分区域账）。"""
    if not os.path.isdir(path):
        return 0
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "*")) if os.path.isfile(p))


def _score_dimensions(issues):
    """按维度扣分：每维从 100 起，ERROR -35 / WARN -12 / INFO -3，下限 0。"""
    scores = {k: 100.0 for k, _, _ in DIMENSIONS}
    for it in issues:
        if it["dim"] in scores:
            scores[it["dim"]] = max(0.0, scores[it["dim"]] - PENALTY.get(it["level"], 5))
    return scores


def _grade(score):
    if score >= 90:
        return "A", "优秀：产物齐全、可复现、可追溯"
    if score >= 75:
        return "B", "良好：存在少量非阻断问题，建议课后修正"
    if score >= 60:
        return "C", "及格：有明显缺项，交付前需整改"
    if score >= 40:
        return "D", "需整改：追溯链不完整，实验结论不可信"
    return "E", "不合格：核心产物缺失或不一致，必须重做"


def collect_report(cfg, logger):
    """执行一次完整巡检，返回结构化报告 dict。"""
    issues = []
    log_dir = os.path.abspath(cfg.log_dir)
    logger.info("==== 数据运维巡检开始 ====")
    logger.info("扫描范围：logs=%s  data=%s  outputs=%s",
                log_dir, os.path.abspath(cfg.data_dir), os.path.abspath(cfg.output_dir))

    source = audit_dataset_source(cfg, issues, logger)
    run_dirs = sorted(d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.isdir(d))
    focus = getattr(cfg, "run_name", None)
    if focus:                       # 显式给了 --run_name 就只查这一个 run
        picked = [d for d in run_dirs if os.path.basename(os.path.normpath(d)) == focus]
        if not picked:
            issues.append(_issue("R3", "completeness", ERROR, os.path.join(log_dir, focus),
                                 "指定的 run 不存在：--run_name %s" % focus,
                                 "去掉 --run_name 可巡检全部 run；或先用 --mode train 跑出该 run"))
        run_dirs = picked
    runs = [scan_run(d, source, issues, cfg) for d in run_dirs]
    runs = [r for r in runs if r.get("files")]
    if not runs:
        issues.append(_issue("R2", "completeness", ERROR, log_dir,
                             "logs/ 下没有任何 run 产物，巡检无对象",
                             "先运行：python main.py --mode train --run_name demo"))

    # 推理成果与 run 的关联（outputs/examples 只有一份，属于全局产物）
    infer_path = os.path.join(os.path.abspath(cfg.output_dir), "examples", "inference_results.json")
    inference = {"exists": os.path.exists(infer_path), "path": infer_path}
    if inference["exists"]:
        data, err = _load_json(infer_path)
        if err:
            issues.append(_issue("N15", "consistency", ERROR, infer_path, f"推理结果损坏：{err}", "重跑推理"))
        else:
            inference.update({"run_name": data.get("run_name"),
                              "num_examples": data.get("num_examples"),
                              "accuracy_on_examples": data.get("accuracy_on_examples")})
            target = next((r["name"] for r in runs if r["name"] == data.get("run_name")), None)
            if not target:
                issues.append(_issue("N16", "traceability", WARN, infer_path,
                                     f"推理结果引用的 run_name={data.get('run_name')} 在 logs/ 中不存在",
                                     "模型权重已被清理或改名，输入图与模型无法配对追溯"))
            elif data.get("accuracy_on_examples") is not None:
                r = next(x for x in runs if x["name"] == target)
                va = r.get("final", {}).get("val_acc")
                if va and abs(data["accuracy_on_examples"] - va) > 0.34:
                    issues.append(_issue("N17", "consistency", WARN, infer_path,
                                         f"示例准确率 {data['accuracy_on_examples']} 与该 run 的 "
                                         f"val_acc {va} 差距过大", "确认推理用的权重与日志是否同一次训练"))
    else:
        issues.append(_issue("C3", "completeness", WARN, infer_path,
                             "尚无推理成果 inference_results.json",
                             "运行：python main.py --mode inference --run_name <run名>"))

    custom = audit_custom_inputs(cfg, runs, issues)

    # 分区域存储账：把项目产物按区域切开，回答「磁盘被哪一类产物吃掉了」
    examples_dir = os.path.join(os.path.abspath(cfg.output_dir), "examples")
    areas = {
        "logs": sum(r["total_bytes"] for r in runs),
        "data": source["total_bytes"],
        "outputs_examples": _dir_bytes(examples_dir),
        "outputs_custom": custom["out_bytes"],
        "inputs": custom["total_bytes"],
        "ops": _dir_bytes(os.path.abspath(cfg.ops_dir)),
    }

    dims = _score_dimensions(issues)
    total = round(sum(dims[k] * w for k, _, w in DIMENSIONS), 1)
    grade, verdict = _grade(total)
    n_err = sum(1 for i in issues if i["level"] == ERROR)
    n_warn = sum(1 for i in issues if i["level"] == WARN)
    n_info = sum(1 for i in issues if i["level"] == INFO)

    storage_by_cat = {c: sum(r["by_cat"][c] for r in runs)
                      for c in ("weights", "logs", "images", "json", "other")}
    best = max((r for r in runs if r.get("final")), key=lambda r: r["final"]["best_val_acc"], default=None)

    report = {
        "meta": {
            "tool": "data_ops.py 数据运维巡检",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": os.path.abspath(os.path.dirname(__file__)),
            "run_name_filter": ("只查 run：%s" % focus) if focus else "全部 run（logs/ 下所有子目录）",
            "log_dir": log_dir, "data_dir": os.path.abspath(cfg.data_dir),
            "output_dir": os.path.abspath(cfg.output_dir), "ops_dir": os.path.abspath(cfg.ops_dir),
            "image_dir": os.path.abspath(os.path.expanduser(getattr(cfg, "image_dir", None) or "./inputs")),
            "retention_days": cfg.retention_days,
        },
        "score": total, "grade": grade, "verdict": verdict,
        "dimensions": dims,
        "summary": {"runs": len(runs), "error": n_err, "warn": n_warn, "info": n_info,
                    "issues": len(issues),
                    "best_run": best["name"] if best else None,
                    "best_val_acc": best["final"]["best_val_acc"] if best else None},
        "runs": runs, "source": source, "inference": inference, "custom": custom,
        "storage": {"runs_bytes": sum(r["total_bytes"] for r in runs),
                    "source_bytes": source["total_bytes"],
                    "areas": areas,
                    "total_bytes": sum(areas.values()),
                    "by_cat": storage_by_cat},
        "issues": sorted(issues, key=lambda i: (LEVEL_RANK[i["level"]], i["dim"], i["code"])),
    }
    logger.info("巡检完成：健康度 %.1f（%s）｜ERROR=%d WARN=%d INFO=%d｜run 数=%d",
                total, grade, n_err, n_warn, n_info, len(runs))
    return report


# ---------------------------------------------------------------- Markdown 报告

def write_markdown_report(rep, path):
    """生成人类可读的巡检报告（可直接贴进实验报告）。"""
    s = rep["summary"]
    L = [
        "# 数据运维巡检报告",
        "",
        f"- 生成时间：{rep['meta']['generated_at']}",
        f"- 巡检范围：`{rep['meta']['log_dir']}` / `{rep['meta']['data_dir']}` / `{rep['meta']['output_dir']}`",
        f"- **健康度：{rep['score']} / 100　等级 {rep['grade']}** —— {rep['verdict']}",
        f"- run 数：{s['runs']}　告警：ERROR {s['error']} / WARN {s['warn']} / INFO {s['info']}",
        f"- 最优 run：`{s['best_run']}`（best_val_acc={s['best_val_acc']}）",
        "",
        "## 1. 五维健康度",
        "",
        "| 维度 | 权重 | 得分 | 评级 |",
        "|---|---|---|---|",
    ]
    for key, label, w in DIMENSIONS:
        sc = rep["dimensions"][key]
        L.append(f"| {label} | {int(w * 100)}% | {sc:.0f} | {_grade(sc)[0]} |")

    L += ["", "## 2. 资产台账（run × 产物）", "",
          "| run | 体积 | 最后更新 | 权重 | 参数 | 过程 | 曲线 | 数据留档 | 日志 |",
          "|---|---|---|---|---|---|---|---|---|"]
    mark = lambda b: "✅" if b else "❌"
    for r in rep["runs"]:
        a = r["artifacts"]
        L.append(f"| `{r['name']}` | {r['human_size']} | {r['age_days']}天前 | "
                 f"{mark(a['model'] and a['best_model'])} | {mark(a['config'])} | {mark(a['history'])} | "
                 f"{mark(a['curves'])} | {mark(a['record'])} | {mark(a['log'])} |")

    L += ["", "## 3. 存储占用", "", "### 3.1 分区域账", "",
          "| 区域 | 占用 | 说明 |", "|---|---|---|"]
    area_desc = {"logs": "训练产物（日志/权重/记录/曲线）", "data": "MNIST 原始数据集",
                 "outputs_examples": "测试集示例输入图 + 识别拼图", "outputs_custom": "用户自提图的预处理结果",
                 "inputs": "用户自提原图（新数据入口）", "ops": "巡检自身报告与看板"}
    for name, val in sorted(rep["storage"]["areas"].items(), key=lambda kv: -kv[1]):
        desc = area_desc.get(name, "")
        real_root = rep["source"].get("root")
        if name == "data" and real_root and \
                os.path.abspath(real_root) != os.path.abspath(rep["meta"]["data_dir"]):
            desc += f"（实际统计自 `{real_root}`，原因见告警 S10）"
        L.append(f"| `{name}` | {human_bytes(val)} | {desc} |")
    L += [f"| **合计** | **{human_bytes(rep['storage']['total_bytes'])}** | |",
          "", "### 3.2 按文件类型", "", "| 类别 | 占用 |", "|---|---|"]
    cat_cn = {"weights": "模型权重(.pdparams)", "logs": "日志(.log)", "images": "图片(.png)",
              "json": "记录(.json)", "other": "其他"}
    for c, v in sorted(rep["storage"]["by_cat"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {cat_cn.get(c, c)} | {human_bytes(v)} |")

    L += ["", "## 4. 训练指标（运维视角）", "",
          "| run | 轮数 | train_acc | val_acc | 泛化间隙 | 最优轮 | best_val_acc |",
          "|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        f = r.get("final") or {}
        if not f:
            L.append(f"| `{r['name']}` | - | - | - | - | - | -（无 history.json，指标不可追溯）|")
            continue
        L.append(f"| `{r['name']}` | {f['epochs']} | {f['train_acc']} | {f['val_acc']} | "
                 f"{f['gap']:+} | {f['best_epoch']} | {f['best_val_acc']} |")
    if rep.get("inference", {}).get("exists"):
        i = rep["inference"]
        L += ["", f"推理成果：run `{i.get('run_name')}`，示例 {i.get('num_examples')} 张，"
                  f"示例准确率 {i.get('accuracy_on_examples')}"]

    src = rep["source"]
    L += ["", "## 5. 数据源体检", "",
          f"- 实际数据目录：`{src['root'] or '未找到'}`",
          f"- 通过 IDX 校验：{src['ok_count']} / {len(MNIST_FILES)}",
          f"- 指纹基线：`{src['fingerprint_path']}`", "",
          "| 文件 | 存在 | 体积 | 样本数 | 尺寸 | IDX | md5 | 基线 |", "|---|---|---|---|---|---|---|---|"]
    for f in src["files"]:
        idx = f.get("idx") or {}
        L.append(f"| `{f['name']}` | {'✅' if f['exists'] else '❌'} | "
                 f"{human_bytes(f.get('size')) if f['exists'] else '-'} | {idx.get('count', '-')} | "
                 f"{'28x28' if idx.get('rows') == 28 else (str(idx.get('rows')) if idx.get('rows') else '-')} | "
                 f"{'✅' if idx.get('ok') else '❌'} | `{(f.get('md5') or '-')[:10]}` | "
                 f"{'⚠️漂移' if f.get('drift') else ('✅' if f.get('baseline_md5') else '-')} |")

    cust = rep.get("custom") or {}
    L += ["", "## 6. 用户自提输入区（新数据接入体检）", "",
          f"- 输入目录：`{cust.get('image_dir')}`（{'存在' if cust.get('dir_exists') else '不存在'}），"
          f"现有 PNG {len(cust.get('pngs') or [])} 张，因格式被排除 {len(cust.get('wrong_ext') or [])} 个",
          f"- 成果：`{cust.get('results_path')}`（{'已生成' if cust.get('exists') else '尚未生成'}）", ""]
    if cust.get("exists"):
        L += [f"- 引用 run：`{cust.get('run_name')}`　模型结构：{cust.get('model_type')}　"
              f"生成于 {cust.get('generated_at')}",
              f"- 记录输入数：{cust.get('num_images')}　成功：{cust.get('succeeded')}　"
              f"预处理失败：{len(cust.get('failed') or [])}　无标注：{cust.get('unlabeled')}　"
              f"标注样本准确率：{cust.get('accuracy')}",
              f"- 权重可达：{'❌ 已丢失 ' + str(cust.get('model_missing')) if cust.get('model_missing') else '✅ 存在'}"
              f"　孤儿产物：{len(cust.get('orphans') or [])} 个", ""]
        if cust.get("failed"):
            L += ["### 6.1 数据质量工单（预处理被拒的图）", "",
                  "| 文件 | 拒因 | 怎么改 |", "|---|---|---|"]
            fix = lambda msg: ("改单字、提高对比度" if "对比度" in msg else
                               "把笔画画粗画黑或重新书写" if "没检测到笔迹" in msg else
                               "只保留一个数字、去掉背景阴影" if "几乎都判成了笔迹" in msg else "见报因")
            for f in cust["failed"]:
                L.append(f"| `{f['file']}` | {f['error']} | {fix(f['error'])} |")
            L.append("")

    L += ["## 7. 告警清单（按级别）", "",
          "| 级别 | 编号 | 维度 | 对象 | 问题 | 建议动作 |", "|---|---|---|---|---|---|"]
    dim_cn = dict((k, lab) for k, lab, _ in DIMENSIONS)
    for it in rep["issues"]:
        L.append(f"| {it['level']} | {it['code']} | {dim_cn.get(it['dim'], it['dim'])} | "
                 f"`{it['target']}` | {it['message']} | {it['suggestion'] or '-'} |")
    if not rep["issues"]:
        L.append("| - | - | - | - | 无告警，全部通过 | - |")

    L += ["", "## 8. 整改优先级", ""]
    todo = [i for i in rep["issues"] if i["level"] == ERROR] + [i for i in rep["issues"] if i["level"] == WARN]
    if not todo:
        L.append("无必须整改项。可继续做参数对比实验。")
    for n, i in enumerate(todo[:8], 1):
        L.append(f"{n}. **[{i['level']}] {i['message']}**  \n   → {i['suggestion'] or '见上文'}")
    L += ["", "> 本报告由 `data_ops.py` 自动生成（只读巡检，不会修改/删除任何产物）。"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------- 巡检入口

def _logger_file(logger):
    """从 logger 的 FileHandler 上取出当前日志文件路径。"""
    for h in getattr(logger, "handlers", []):
        path = getattr(h, "baseFilename", None)
        if path:
            return path
    return None


def run_ops_audit(cfg, logger=None):
    """执行一次完整巡检并落盘全部成果，返回 (report, ops_dir)。

    产出：ops_report.json / ops_report.md / dashboard.html / dashboard.png（可选）
    以及巡检自身日志 outputs/ops/audit/ops_*.log。
    """
    own_logger = logger is None
    log_path = None
    if own_logger:
        from logger import get_logger
        logger, log_path = get_logger(cfg.ops_dir, "audit", prefix="ops")
    else:
        log_path = _logger_file(logger)

    report = collect_report(cfg, logger)
    ops_dir = os.path.abspath(cfg.ops_dir)
    os.makedirs(ops_dir, exist_ok=True)

    json_path = os.path.join(ops_dir, "ops_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = write_markdown_report(report, os.path.join(ops_dir, "ops_report.md"))
    html_path = os.path.join(ops_dir, "dashboard.html")
    from ops_dashboard import render_html_dashboard
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html_dashboard(report))

    # PNG 看板交给 visualize.py 统一出图；无 matplotlib 时自动跳过，不影响其他成果
    png_path = os.path.join(ops_dir, "dashboard.png")
    try:
        from visualize import plot_ops_dashboard
        made = plot_ops_dashboard(report, png_path)
    except Exception as e:
        logger.info("[提示] 未能生成 PNG 看板（%s：%s），HTML/Markdown 看板不受影响",
                    type(e).__name__, e)
        made = None

    for it in report["issues"]:
        if it["level"] == ERROR:
            logger.error("[%s] %s | %s", it["code"], it["target"], it["message"])
    logger.info("健康度 %.1f/100（%s）—— %s", report["score"], report["grade"], report["verdict"])

    logger.info("==== 巡检成果保存位置（绝对路径）====")
    artifacts = [("结构化报告", json_path), ("可读巡检报告", md_path),
                 ("交互看板（浏览器打开）", html_path), ("四联看板图", made),
                 ("本次巡检日志", log_path), ("数据指纹基线", report["source"]["fingerprint_path"])]
    for label, p in artifacts:
        if p:
            logger.info("  %s -> %s", label, os.path.abspath(p))
    if own_logger:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)
    return report, ops_dir
