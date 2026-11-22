# Leonida — Guia rápido (`pt-BR`)

> [!NOTE]
> A documentação em inglês ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) e
> [`docs/README.md`](../../README.md)) é a **referência oficial**. Este guia reúne apenas instruções estáveis de
> configuração e compilação (sincronizado com o commit `003e7812`). Para o status diário do projeto, consulte o
> [`README.md`](../../../README.md).

## O que é o Leonida

O Leonida é um projeto de recompilação estática do Grand Theft Auto VI (Xbox Series) para rodar nativamente no
Windows, construído **no seu próprio PC a partir da sua própria cópia**. Os shaders RDNA2 do console são
traduzidos para SPIR-V (Vulkan), os assets são lidos diretamente do seu pacote e o software de sistema do
console é reimplementado em shims.

> [!IMPORTANT]
> Este repositório **não contém o jogo, chaves nem executável**, e não explica como obtê-los. Nada aqui
> descriptografa ou contorna proteções. Tudo o que o build gera (shaders traduzidos, assets extraídos, pasta de
> trabalho e saves) vem da sua cópia e **deve ficar apenas na sua máquina** — nunca anexe nem compartilhe esses
> arquivos em issues ou pull requests (veja [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

O projeto está em fase inicial: **o jogo ainda não dá boot**.

## Pré-requisitos

- **Sua própria cópia** do pacote Xbox Series, versão `1.0.2.1` (pasta com `MicrosoftGame.Config` e os arquivos `.rpf` legíveis em disco).
- **Windows 10/11 x64** com cerca de **350 GB livres** para arquivos intermediários (podem ficar em outro disco usando `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (marque *"Add python.exe to PATH"*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` e `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) com o **target AMDGPU** (`llvm-mc`):
  - O binário oficial do LLVM para Windows **não** inclui o target AMDGPU.
  - Para compilar um `llvm-mc.exe` nativo com AMDGPU, rode este comando a partir da raiz do repositório.

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

    O script requer `git` e as ferramentas C++ do Visual Studio, configura `LLVM_MC` para o seu usuário
    e pede que você abra um novo terminal para carregar a variável.
  - Se usar o WSL, instale `llvm-21` no Ubuntu e defina a variável na mesma sessão do PowerShell em que vai
    executar `make_game.py`.

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Veja [`docs/design.md`](../../design.md#disassembling-shaders) para mais detalhes.
  - No Linux, basta ter `llvm-mc` no `PATH`.
- **Visual Studio 2022** com *Desenvolvimento para desktop com C++* e **Windows SDK `10.0.26100` ou mais recente** (necessário para o estágio `runtime`, que usa root signatures 1.2).
- [ffmpeg](https://ffmpeg.org/download.html) no `PATH` *(opcional: decodifica streams XMA2 e é exigido ao usar `--wav`)*.

## Como compilar

### Modo simples (Windows)

1. Instale os pré-requisitos acima.
2. Clone o repositório **com o submódulo**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
3. Dê duplo clique em **`Build.bat`**.

O script procura automaticamente sua cópia (Downloads, Desktop, Documentos, Games, pasta de usuário e raiz dos
discos até dois níveis), valida os pré-requisitos e o espaço livre e grava o resultado em `Documents\Leonida`.
Se o build for interrompido, basta rodar novamente para retomar de onde parou.

### Pelo terminal (`make_game.py`)

```powershell
python -I make_game.py --find                          # lista as cópias encontradas neste PC
python -I make_game.py --check                         # verifica os arquivos do jogo sem compilar
python -I make_game.py --game "D:\caminho\do\pacote" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # roda apenas os estágios escolhidos
python -I make_game.py --game "D:\caminho\do\pacote" --wav  # também decodifica Ogg Opus para PCM WAV (grande)
python -I make_game.py --wave64-emulate auto           # auto (padrão), on ou off
```

Os estágios executam nesta ordem: `check` → `shaders` → `assets` → `runtime` → `pipelines`.

- **`--wave64-emulate`**: emula shaders compute/mesh wave64 em GPUs com subgrupos de 32 lanes (NVIDIA e Intel). O padrão `auto` consulta `vulkaninfo` e ativa a emulação automaticamente quando necessário.
- **`--work`**: recomenda-se apontar para um disco com pelo menos 350 GB livres; o estágio `shaders` é o mais demorado e usa metade das threads da CPU por padrão (ajustável com `--jobs`).
- **Linux e Wine/Proton**: `make_game.py` também roda no Linux. Para o estágio `runtime` fora do Windows, passe `--runtime-build <pasta>` com um build Windows (`gta6rt.exe` e `gta6rt_tls_host.dll`), e o script gera `gta6rt.sh` para iniciar via `wine` (ou pela variável `WINE` apontando para o Proton).

## Problemas comuns

- **Baixou pelo botão "Download ZIP" do GitHub**: o ZIP não inclui o submódulo `third_party/rpf9-explorer`. Use `git clone --recursive` ou execute `git submodule update --init --recursive` dentro da pasta do repositório.
- **`llvm-mc` recusa `gfx1013` / AMDGPU**: você provavelmente está com o instalador oficial do LLVM para Windows. Siga um dos passos de configuração do LLVM acima.
- **Erro de `D3D12_ROOT_SIGNATURE_DESC2` ao compilar o runtime**: atualize o Windows SDK para `10.0.26100` ou mais recente pelo Visual Studio Installer ou `winget install Microsoft.WindowsSDK.10.0.26100`, apague a pasta de build e reconfigure.
- **Falha por falta de espaço no drive `C:`**: passe `--work "D:\leonida-work"` para mover os arquivos temporários para outra partição.

## Como contribuir e rodar os testes de GPU

1. Consulte os PRs abertos e a [issue de coordenação][coord] para evitar trabalho duplicado. Reserve a
   tarefa com `claiming: <ID>` ou com uma descrição de uma linha, se ainda não houver ID.
2. Rode os testes relevantes com entradas sintéticas. Por exemplo, para testar o verificador de manifestos,
   execute `python -I tests/test_manifest_verifier.py` na raiz do repositório.
3. No PR, inclua o ID da tarefa, descreva o que testou e informe a GPU e o driver se rodou testes de GPU.
   Confira as regras em [`CONTRIBUTING.md`](../../../CONTRIBUTING.md). Se usou assistência de IA, indique
   isso no PR, conforme [`AGENTS.md`](../../../AGENTS.md).

O [guia de testes em GPU](../../testing.md) mostra como executar os testes sem arquivos do jogo.
Publique os resultados de GPU na [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
