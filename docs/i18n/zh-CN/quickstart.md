# Leonida — 快速上手（`zh-CN`）

> [!NOTE]
> 英文文档（[`README.md`](../../../README.md)、[`CONTRIBUTING.md`](../../../CONTRIBUTING.md) 和
> [`docs/README.md`](../../README.md)）是**权威版本**。本文只汇总稳定的安装与构建说明（与 commit `0d61c357` 同步）。
> 项目每日进展请查看 [`README.md`](../../../README.md)。

## Leonida 是什么

Leonida 是一个把 Xbox Series 版 Grand Theft Auto VI **静态重编译**到 Windows 的项目：在**你自己的电脑上、
用你自己的游戏拷贝**构建。主机的 RDNA2 着色器翻译为 SPIR-V（Vulkan），资源直接从你的包读取，
主机的系统软件用 shim 重新实现。

> [!IMPORTANT]
> 本仓库**不包含**游戏、密钥或可执行文件，也不说明如何获取。这里没有任何解密或绕过保护的内容。构建产生的一切
> （翻译后的着色器、导出的资源、工作目录和存档）都来自你自己的拷贝，**只能留在本机**——不要在 issue 或
> pull request 中附带或分享这些文件（见 [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)）。

项目处于早期：**目前还不能启动游戏**。

## 前置条件

- **你自己的拷贝**：Xbox Series 包，版本 `1.0.2.1`（目录里有 `MicrosoftGame.Config` 和明文可读的 `.rpf` 文件）。
- **Windows 10/11 x64**，约 **350 GB** 空闲空间给中间文件（可用 `--work` 放到另一块盘）。
- [Python 3.12+](https://www.python.org/downloads/)（勾选 “Add python.exe to PATH”）。
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)。
- [Vulkan SDK](https://vulkan.lunarg.com/)（`spirv-as`、`spirv-val`、`vulkaninfo`）。
- [LLVM 21+](https://github.com/llvm/llvm-project/releases)，需要 **AMDGPU target**（`llvm-mc`）：
  - Windows 官方 LLVM 发行版**不含** AMDGPU target。
  - Windows 上运行 `scripts\build_llvm_mc.ps1` 构建原生 `llvm-mc.exe`（需要 `git` 和 Visual Studio 的 C++ 工具）
    并设置 `LLVM_MC`；**或**在 WSL 里安装 `llvm-21` 并设置
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`（见 [`docs/design.md`](../../design.md#disassembling-shaders)）。
  - Linux 上 `PATH` 里有 `llvm-mc` 即可。
- **Visual Studio 2022**，勾选“使用 C++ 的桌面开发”和 **Windows SDK `10.0.26100` 或更新**
  （`runtime` 阶段需要，它使用 root signature 1.2）。
- [ffmpeg](https://ffmpeg.org/download.html) 在 `PATH` 中（可选：解码 XMA2 流；使用 `--wav` 时必需）。

## 如何构建

### 简单方式（Windows）

1. 安装上面的前置条件。
2. **带子模块**克隆仓库：

   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```

3. 双击 **`Build.bat`**。

脚本会自动查找你的拷贝（Downloads、Desktop、Documents、Games、用户目录以及每个盘的根目录往下两层），
检查前置条件和可用空间，结果写到 `Documents\Leonida`。构建中断后重新运行即可从断点继续。

### 命令行（`make_game.py`）

```powershell
python -I make_game.py --find                             # 列出本机找到的拷贝
python -I make_game.py --check                            # 校验游戏文件，不构建
python -I make_game.py --game "D:\path\to\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets            # 只跑指定阶段
python -I make_game.py --game "D:\path\to\package" --wav  # 额外把 Ogg Opus 解码为 PCM WAV（很大）
python -I make_game.py --wave64-emulate auto              # auto（默认）、on 或 off
```

阶段顺序：`check` → `shaders` → `assets` → `runtime` → `pipelines`。

- **`--wave64-emulate`**：在 32 lane 子组的 GPU（NVIDIA、Intel）上模拟 compute/mesh 的 wave64。
  默认 `auto` 会询问 `vulkaninfo`，需要时自动启用。
- **`--work`**：建议指向至少有 350 GB 空闲的磁盘；`shaders` 阶段最慢，默认使用一半 CPU 线程（`--jobs` 可调）。
- **Linux 与 Wine/Proton**：`make_game.py` 也能在 Linux 上运行。非 Windows 的 `runtime` 阶段需要传入
  `--runtime-build <目录>`（含 Windows 构建的 `gta6rt.exe` 和 `gta6rt_tls_host.dll`），脚本会生成
  `gta6rt.sh`，用 `wine`（或 `WINE` 指定的 Proton）启动。

## 常见问题

- **用 GitHub 的 “Download ZIP” 下载**：ZIP 不含 `third_party/rpf9-explorer` 子模块。请用
  `git clone --recursive`，或在仓库目录里执行 `git submodule update --init --recursive`。
- **`llvm-mc` 拒绝 `gfx1013` / AMDGPU**：多半是用了 Windows 官方 LLVM 安装包。改用
  `scripts\build_llvm_mc.ps1`，或把 `LLVM_MC` 指向 WSL 里的 `llvm-mc`（`llvm-21`）。
- **构建 runtime 时报 `D3D12_ROOT_SIGNATURE_DESC2` 错误**：把 Windows SDK 升级到 `10.0.26100` 或更新
  （可在 Visual Studio Installer 或 `winget install Microsoft.WindowsSDK.10.0.26100` 中安装），
  删除构建目录后重新配置。
- **`C:` 盘空间不足**：用 `--work "D:\leonida-work"` 把中间文件放到其他分区。

## 参与贡献与运行 GPU 测试

- GPU 测试指南（不需要游戏文件）：[`docs/testing.md`](../../testing.md)。
- 贡献规则、零游戏数据政策与任务协调：[`CONTRIBUTING.md`](../../../CONTRIBUTING.md)。
