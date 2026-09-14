---
name: import-local-model
description: >
  把本地 .safetensors 文件导入 Alis Studio：模型 checkpoint（检查 MLX 打包量化 /
  ComfyUI int8_convrot 旋转量化 / bf16，需要时用 scripts/convert_krea2_official.py 做含
  Hadamard 去旋转的转换，再注册为 Krea 2 后端的本地模型）和 LoRA（检查 base 模型与
  LoKr/低秩键格式，放进 LoRA 库目录即可用）。当用户要求"添加/导入本地模型或 LoRA"、
  "转换模型文件"、"模型/LoRA 出噪点/出乱图/加载失败/不生效"时使用。
---

# 导入本地模型文件

本项目的模型权重由 `krea2-alis-mlx` 包 strict 加载，只认三种精度：`8bit`、`mixed-4-8`、`bf16`，
布局是 MLX 打包量化（U32 weight + group-64 的 `scales`/`biases`）。Civitai/社区下载的文件
往往是别的格式，**直接注册会加载成功但生成乱图**——必须先检测格式。不要修改 venv 里的
krea2 包，所有处理都在应用层完成。

**两类文件走不同流程，先分清**：LoRA 是小文件（几十 MB – 2 GB，键名带 `lora_`/`lokr_`），
放进 LoRA 库目录即可，见文末[LoRA 导入](#lora-导入)；checkpoint 是完整模型（8–15 GB，
键名是 `blocks.*.weight`/`scales` 等结构张量），走下面的完整流程。

## 第一步：检测原始格式

用 safetensors 头判断（只读头部几 MB，不要加载权重）。项目里已有现成工具：

```bash
venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, ".")
from studio import local_models as lm
h = lm._read_header("<文件路径>")
print("int8_convrot 旋转量化:", lm._is_row_scale_repackage(h))
print("推断 MLX 精度:", lm._infer_precision(h))   # 8bit / mixed-4-8 / bf16
import json
print("metadata:", json.dumps(h.get("__metadata__", {}), indent=1)[:800])
EOF
```

判定规则：

| 特征 | 格式 | 处理 |
|---|---|---|
| 有 `*.scales`（U32 weight 配对） | MLX 打包量化 | 直接注册，精度用 `_infer_precision` 的结果 |
| I8 权重 + `*.weight_scale`，metadata 里 `target_format: int8_convrot` | **ComfyUI ConvRot 旋转量化** | 必须先走第二步转换，**直接注册会出乱图** |
| I8 + weight_scale 但无 `_quantization_metadata` | 未知的行量化格式 | 不要猜；报告用户并停止 |
| 无量化张量 | bf16 | 直接注册 |

ConvRot 的原理：每层的 `W_rot = W @ Hᵀ`（沿输入维每 `convrot_groupsize` 一组、块对角
"regular" Hadamard——4×4 核 `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` 的 Kronecker 幂，
归一化 1/√n），推理时激活值同样旋转以抵消。MLX 管线不知道旋转 → 计算全错。
**哪些层被旋转以文件自带 `_quantization_metadata` 的 per-layer 标记为准**——DiT 主体
`blocks.*` 和文本条件路径 `txtfusion.*` 都可能是旋转的，不要按键名前缀猜。

## 第二步：转换（仅 int8_convrot 需要）

```bash
venv/bin/python scripts/convert_krea2_official.py <原始文件路径>
```

- 输出写到旁边：`<stem>.mlx8bit.safetensors`（约 14 GB，磁盘要留够）
- 耗时几分钟，峰值内存 ~1 GB（流式处理），进度每 32 个张量刷新一次
- 脚本做的事：反量化（I8×行 scale）→ 对 metadata 标记的层做 Hadamard 去旋转（以
  metadata 为准，覆盖 blocks.* 和 txtfusion.*）→ 主体层 `mx.quantize(group_size=64,
  bits=8)` 重量化、其余转 bf16 → 手写 safetensors 流式写出

### 转换后必须做全量验证（跳过这步会交付坏权重）

1. **全量自洽性**：全部 430 个输出张量与"源反量化+去旋转真值"逐个比 cos，必须 0 个
   mismatch（cos < 0.99 即失败）。**不要抽查几个张量就下结论**——曾因漏掉 txtfusion
   32 层翻车：抽查全过、生成却是"规律几何形状"。
2. **strict 加载**：

```bash
venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "."); sys.path.insert(0, "venv/lib/python3.12/site-packages")
import mlx.core as mx
from mlx import nn
from krea2.transformer import SingleStreamDiT, Krea2Config
from krea2.quant_recipes import quantize_bulk
m = SingleStreamDiT(Krea2Config())
nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
m.load_weights("<转换后的文件>", strict=True)
print("strict load OK")
EOF
```

3. （可选）与官方基线做数值对照：本机缓存
   `~/.cache/krea2_alis_mlx/avlp12__Krea-2-Turbo-Alis-MLX-mixed-4-8/` 有对照权重，去旋转后
   同层 cos 应 >0.99；若 ~0.06 说明去旋转没生效。

### 症状速查（历史上踩过的坑）

| 生成结果 | 根因 |
|---|---|
| 纯雪花噪点 | 旋转层全部没去旋转（W_rot 被当 W 用） |
| 不规律的几何形状、有结构不成图 | 部分层没去旋转（如只修了 blocks.* 漏了 txtfusion.*） |

## 第三步：注册进模型列表

**先把模型文件放到固定位置再注册**——推荐 `~/models/`。模型是原地引用的，注册表里记录的就是
这个路径；不要把文件留在 `~/Downloads/` 注册（清理工具容易扫走它，之后就是"文件缺失"）。
如果文件已在 Downloads：先 `mv` 到 `~/models/`，再注册新路径。文件日后挪动后同样要
删除旧条目、按新路径重新注册。

```bash
mkdir -p ~/models && cp /path/to/model.mlx8bit.safetensors ~/models/   # 转换产物放进固定目录
venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from studio import local_models as lm
print(lm.add('/Users/$USER/models/<文件名>.safetensors'))   # 读同名旁车 json（可选 {precision,label,min_ram}），自动推断精度
"
```

- 文件**原地引用**不复制；注册表在 `~/Library/Application Support/Alis Studio/local_models.json`
- 验证：`curl -s localhost:<port>/api/catalog` 里 krea2-turbo 出现
  `local:<name>` 且 `installed: true`
- 删除用 UI 垃圾桶或 `lm.remove(name)`，只取消注册、不删文件

## 重启时机（重要）

- 模型**列表**：动态读注册表，运行中的服务器立即可见，无需重启。
- 模型**权重**：首次生成后管线常驻内存，`will_load()` 只看 variant 是否切换。
  **替换/重转文件后必须重启 app**，否则内存里还是旧权重，新生成仍然用坏数据。

## 注意

- **模型文件的常驻位置是 `~/models/`**：注册引用的是绝对路径，文件挪走/被清理 = 模型缺失。
  转换完成后把产物从 Downloads 移入 `~/models/` 再注册；原始下载文件（int8_convrot 源）转换
  验证通过后可以删除，需要时再从 Civitai 重新下载。
- 8-bit 模型 ~14 GB，RAM 门槛 24 GB；用户 Mac 内存不足时提醒换小模型或调低分辨率
- 想覆盖推断结果/改显示名：在模型文件旁放同名 `.json`（如 `{"precision": "8bit", "label": "My Mix"}`）
- 精度推断失败时报错并停止，不要瞎猜

---

# LoRA 导入

LoRA **没有注册表——库目录即列表**：`~/Library/Application Support/Alis Studio/loras/` 下的
`.safetensors` 文件就是全部条目，运行中的服务实时可见（`/api/loras` 每次现扫），**无需重启**、
无需转换（两种主流键格式后端都原生支持）。UI 的 LoRA 面板渲染时会自动重读目录。

## 第一步：检测（只读 header，别加载权重）

```bash
venv/bin/python - <<'EOF'
import json, struct
path = "<文件路径>"
with open(path, "rb") as f:                       # 自包含解析：lm._read_header 会丢弃 __metadata__，
    (n,) = struct.unpack("<Q", f.read(8))         # 而 base 模型判定恰恰全靠它，所以这里手读
    header = json.loads(f.read(n))
keys = [k for k in header if k != "__metadata__"]
meta = header.get("__metadata__") or {}
print("base:", meta.get("ss_base_model_version"), "| trained by:",
      (json.loads(meta["software"])["name"] if "software" in meta else "?"))
kinds = {}
for suf in (".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
            ".lokr_t2", ".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight",
            ".alpha"):
    c = sum(1 for k in keys if k.endswith(suf))
    if c: kinds[suf] = c
print("key kinds:", kinds, "| tensors:", len(keys))
EOF
```

判定规则（全部满足才导入）：

| 检测结果 | 结论 | 处理 |
|---|---|---|
| `ss_base_model_version` 含 `krea2`，键为 `*.lokr_w1` + `*.lokr_w2`（可带 `.alpha`） | ai-toolkit **LoKr**（Krea 2 社区 LoRA 的主流格式） | 直接导入；`.alpha` 缓冲被有意忽略（ai-toolkit 与 ComfyUI 对直接 w1/w2 都按 scale 1.0 应用） |
| base 为 krea2，键为 `*.lora_down/up` 或 `*.lora_A/B`（配 `alpha`/metadata 的 `lora_alpha`+`lora_rank`） | 普通**低秩 LoRA** | 直接导入（`krea2-alis-mlx` 的运行时低秩分支） |
| base 为 krea2，但有 `*.lokr_w1_a/_b`、`*.lokr_w2_a/_b`、`*.lokr_t2` | **低秩因子 LoKr**（少见） | 不支持，明确告知用户；不要导入（生成时会报错） |
| base 不是 krea2（`sd15`/`sdxl`/`flux` 等）或无 base 元数据 | **别的模型的 LoRA** | 拒绝导入，告知需要 Krea 2 专训版本；硬导入会在生成时报 "won't fit" |
| 键是 `blocks.*.weight`/`scales`/`biases` 等结构张量，文件 >5 GB | 这不是 LoRA，是 **checkpoint** | 走上面的模型导入流程 |

## 第二步：放进库目录

```bash
mkdir -p ~/Library/Application\ Support/Alis\ Studio/loras
mv /path/to/<lora>.safetensors ~/Library/Application\ Support/Alis\ Studio/loras/
```

- 文件名即库内显示名（去掉 `.safetensors` 后缀）。合法字符：字母数字、`-`、`_`、`.`、空格；
  不以 `.` 开头。不合法的名字 UI 删不掉、生成时也解析不到（server 的 `_safe_lora_name` 会拒）。
- 放 Downloads 里直接引用**不行**（LoRA 是复制进库的，不像 checkpoint 原地引用），必须移入/拷入库目录。
- 覆盖同名文件前先确认用户（已保存的生图配方按名字引用 LoRA，静默替换会让旧配方的含义改变）。

## 第三步：验证

1. **列表可见**：`curl -s localhost:7860/api/loras`（或实际端口）出现新条目即成功，无需重启。
2. **效果对比**（推荐，同 seed）：先不带 LoRA 生成一张基线，再勾选 LoRA 生成同 prompt/seed 的一张——
   两图应明显不同；然后取消勾选再生成一次，应与基线**逐位一致**（md5 相同，运行时分支卸载是精确恢复）。
3. **症状速查**：

| 现象 | 根因 |
|---|---|
| 生成时报 "won't fit"/"doesn't exist in the model tree"/"shape mismatch" | LoRA 不是给 Krea 2 训的（base 不匹配），换文件 |
| 生成时报 "low-rank (_a/_b tensors)" | 低秩因子 LoKr，暂不支持 |
| 生成报 "Unrecognized LoRA key" | 未知键格式（`krea2/lora.py` 的 `_normalize` 只认低秩对；LoKr 由后端 `studio/backends/krea2.py` 处理）——把报错原样报告 |
| 图像带规律的几何伪影、局部错乱 | LoRA 强度过高或文件损坏；先降 strength 到 0.5 复测，再考虑重下 |
| 勾选后效果毫无变化 | 检查是否真的勾选 + scale>0；同 seed 基线对比没做对（seed 不同时"没变化"不可信） |
