# End-to-end Web Demo

本 web demo 是参考伙伴项目 `English_assessment_project` 的前端交互补充到本项目中的版本，但后端已经改为调用本项目自己的端到端 pipeline。

## 启动

```powershell
python webapp\app.py --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

## 功能

- 浏览器录音或上传音频。
- 输入目标英文文本。
- 自动音频预处理到 16 kHz mono PCM wav。
- G2P 文本转音素。
- 音频与目标音素对齐。
- 音素级发音诊断。
- 显示四类结果：
  - `correct`
  - `acceptable_accent`
  - `true_error`
  - `uncertain_review`
- 显示 `alignment_quality` 和 `review_reason`。
- 诊断产物自动保存到 `outputs/webapp/`。

## 与伙伴项目前端的主要区别

伙伴项目 webapp 调用的是 `run_minimal_phone_demo.py`，文档中说明当前边界分配使用 uniform segmentation fallback。

本项目版本调用的是：

```text
audio preprocess
-> G2P
-> alignment
-> verifier prediction
-> manual calibration
-> decision rule
```

如果 alignment 失败，本项目不会返回空表，而是基于 G2P 生成 fallback rows，并强制输出：

```text
alignment_quality = bad
decision = uncertain_review
confidence = 0.0
```

这样可以避免把文本/音频不一致或对齐失败误判为真实发音错误。

