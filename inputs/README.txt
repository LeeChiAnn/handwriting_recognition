本目录是「自定义推理」的输入区（--mode custom 读取这里，其他模式不看这里）。

放什么
  - 一张 PNG 只写一个数字，0-9 任意；张数不限，按文件名顺序处理。
  - 尺寸不限：28×28 的小图、手机拍的 4000×3000 大图都能直接放，
    程序会自动：灰度化 → 判断黑白极性（白底黑字会帮你反色）→ Otsu 自动阈值
    分离笔迹 → 裁外接框 → 等比缩放到长边 20px → 按灰度质心居中到 28×28。
    预处理细节见 custom_input.py。
  - 建议：笔画黑、背景白、别带手指/桌面阴影/格线；数字写大写满一点，识别更稳。

文件名规则（可选，用来算准确率）
  - 以数字开头 → 该数字被当作标准答案，例如 `7.png`、`3_我的写法.png`。
  - 不以数字开头（如 `my.png`）→ 只输出预测结果，不并入准确率统计。

两步跑通
  1) 先自检图片可用（不需要 paddle，只打印诊断，不写任何文件）：
       python custom_input.py inputs
  2) 再识别（需要先有训练好的模型，run_name 用你训练时那个）：
       python main.py --mode train  --run_name demo --epochs 5
       python main.py --mode custom --run_name demo --image_dir inputs

产物在 outputs/custom/（不会改动 --mode inference 的 outputs/examples/）
  - sample_N_predX.png        模型真正看到的 28×28 图（文件名带 true 说明你标了答案）
  - predictions_custom.png    识别拼图：pred / 置信度 p / 对错
  - custom_results.json       结构化结果：预测、top3、置信度、原图尺寸、预处理诊断

没有图想先试试？把预览图复制过来当输入即可（占位字形，不是真实手写）：
  cp outputs/examples/preview/digit_7.png inputs/7.png
