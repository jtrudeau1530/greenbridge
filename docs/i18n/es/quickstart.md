# Leonida — Guía rápida (`es`)

> [!NOTE]
> La documentación en inglés ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) y
> [`docs/README.md`](../../README.md)) es la **referencia oficial**. Esta guía reúne únicamente instrucciones estables
> de configuración y compilación (sincronizado con el commit `4eb6155d`). Para consultar el estado del proyecto, que
> cambia a diario, consulta el [`README.md`](../../../README.md).

## Qué es Leonida

Leonida es un proyecto de recompilación estática de Grand Theft Auto VI (versión Xbox Series) para ejecutarse
nativamente en Windows, construido **en tu propio PC a partir de tu propia copia del juego**. Los shaders RDNA2 de la
consola se traducen a SPIR-V (Vulkan), los recursos (assets) se leen directamente desde tu paquete y el software de
sistema de la consola se reimplementa mediante shims.

> [!IMPORTANT]
> Este repositorio **no contiene el juego, claves ni ejecutables**, y no explica cómo obtenerlos. Nada aquí descifra
> ni elude ningún tipo de protección. Todo lo que genera la compilación (shaders traducidos, recursos extraídos,
> carpeta de trabajo y partidas guardadas) proviene de tu copia y **debe permanecer únicamente en tu máquina**: nunca
> adjuntes ni compartas estos archivos en issues o pull requests (consulta
> [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

El proyecto se encuentra en una fase temprana: **el juego aún no arranca**.

## Requisitos previos

- **Tu propia copia** del paquete Xbox Series, versión `1.0.2.1` (carpeta que contiene `MicrosoftGame.Config` y los
  archivos `.rpf` legibles en disco).
- **Windows 10/11 x64** con aproximadamente **350 GB libres** para archivos temporales de trabajo (se pueden ubicar en
  otra unidad usando `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (marca la casilla *"Add python.exe to PATH"* durante la instalación).
- [SDK .NET 8](https://dotnet.microsoft.com/download/dotnet/8.0).
- [SDK Vulkan](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` y `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) con el **objetivo AMDGPU** (`llvm-mc`):
  - El binario oficial de LLVM para Windows **no incluye** el objetivo AMDGPU.
  - Para compilar un `llvm-mc.exe` nativo con AMDGPU, ejecuta desde la raíz del repositorio:

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

    El script requiere `git` y las herramientas de C++ de Visual Studio; además, configura `LLVM_MC` para tu
    usuario y te pedirá abrir un nuevo terminal para cargar la variable de entorno.
  - Si usas WSL, instala `llvm-21` en Ubuntu y define la variable en la misma sesión de PowerShell donde vayas a
    ejecutar `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Consulta [`docs/design.md`](../../design.md#disassembling-shaders) para más detalles.
  - En Linux, basta con tener `llvm-mc` accesible en el `PATH`.
- **Visual Studio 2022** con la carga de trabajo *Desarrollo para el escritorio con C++* y el **SDK de Windows `10.0.26100`
  o posterior** (necesario para la etapa `runtime`, que utiliza root signatures 1.2 no presentes en SDKs anteriores).
- [ffmpeg](https://ffmpeg.org/download.html) en el `PATH` *(opcional: decodifica los flujos de audio XMA2, y obligatorio
  si se usa `--wav`)*.

## Cómo compilar

### Modo sencillo (Windows)

1. Instala los requisitos previos anteriores.
2. Clona el repositorio **con su submódulo**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
3. Haz doble clic en **`Build.bat`**.

El script localiza tu copia automáticamente: busca en Descargas, Escritorio, Documentos, Juegos, tu carpeta de usuario
y en la raíz de cada unidad de disco (hasta dos niveles de profundidad). Si encuentra varias copias, te preguntará cuál
usar; si no encuentra ninguna, te solicitará la ruta de la carpeta. Antes de comenzar, comprueba los requisitos previos y
el espacio libre disponible, e informa de golpe de cualquier elemento que falte. El resultado se guarda en
`Documents\Leonida`. Si la compilación se interrumpe, continuará desde donde se quedó al volver a ejecutarla.

### Desde el terminal (`make_game.py`)

```powershell
python -I make_game.py --find                          # lista las copias encontradas en este PC
python -I make_game.py --check                         # verifica los archivos del juego sin compilar nada
python -I make_game.py --game "D:\ruta\a\tu\paquete" --out "D:\Juegos\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # ejecuta únicamente las etapas indicadas
python -I make_game.py --game "D:\ruta\a\tu\paquete" --wav  # decodifica también Opus a PCM WAV (mayor tamaño)
python -I make_game.py --wave64-emulate auto           # auto (predeterminado), on u off
```

Las etapas se ejecutan en este orden: `check` → `shaders` → `assets` → `runtime` → `pipelines`.

- **`--wave64-emulate`**: traduce los shaders de compute y mesh wave64 para que sean exactos independientemente del
  tamaño de subgrupo de la GPU. Las GPUs NVIDIA e Intel tienen subgrupos de 32 hilos y lo requieren. El valor por
  defecto, `auto`, consulta `vulkaninfo` y activa la emulación automáticamente salvo que todas las GPUs reporten
  subgrupos de 64 hilos.
- **`--work`**: se recomienda indicar un disco con al menos 350 GB libres. La etapa de traducción de shaders es la más
  lenta y utiliza por defecto la mitad de los hilos de la CPU (ajustable con `--jobs`).
- **`--wav`**: desactivado por defecto, ya que el audio PCM WAV de 16 bits ocupa varias veces más espacio que Ogg Opus.
- **Linux y Wine/Proton**: `make_game.py` también funciona en Linux. El cargador es un ejecutable de Windows y se ejecuta
  bajo Wine o Proton. Para la etapa `runtime`, puedes pasar `--runtime-build <carpeta>` con una compilación previa de
  Windows (`gta6rt.exe` y `gta6rt_tls_host.dll`); el script generará `gta6rt.sh`, que inicia `gta6rt.exe` con `wine` (o
  con el Wine indicado mediante la variable `WINE`, como `files/bin/wine` en Proton).

## Problemas comunes

- **Descargado mediante "Download ZIP" desde GitHub**: el archivo ZIP no incluye el submódulo
  `third_party/rpf9-explorer`. Usa `git clone --recursive` o ejecuta `git submodule update --init --recursive` en la raíz
  del repositorio.
- **`llvm-mc` rechaza `gfx1013` o el objetivo AMDGPU**: probablemente estés usando el instalador oficial de LLVM para
  Windows. Ejecuta `scripts\build_llvm_mc.ps1` o apunta `LLVM_MC` al `llvm-mc` de WSL (`llvm-21`).
- **Error en CMake o MSVC sobre `10.0.26100` o `D3D12_VERSIONED_ROOT_SIGNATURE_DESC`**: actualiza el Windows SDK a
  `10.0.26100` o superior desde el Visual Studio Installer o mediante `winget install Microsoft.WindowsSDK.10.0.26100`.
  Posteriormente, elimina la carpeta `runtime/build` y vuelve a configurar.
- **Falta de espacio en la unidad `C:`**: usa `--work "D:\leonida-work"` para trasladar los archivos temporales a otra
  partición o disco.
- **Windows bloquea los ejecutables de prueba recién compilados** («Una directiva de Control de aplicaciones ha bloqueado
  este archivo»): se debe al Control de aplicaciones inteligente de Windows 11. Consulta
  [`docs/testing.md`](../../testing.md#known-problems).

## Cómo contribuir y ejecutar las pruebas de GPU

1. Consulta los Pull Requests abiertos y la [issue de coordinación][coord] para no duplicar esfuerzos. Reserva la tarea
   comentando `claiming: <ID>` (o con una descripción breve de una línea si aún no tiene identificador).
2. Ejecuta las pruebas pertinentes con entradas sintéticas antes de enviar cambios. Por ejemplo, para verificar el
   comprobador de manifiestos: `python -I tests/test_manifest_verifier.py`.
3. En tu Pull Request, incluye el identificador de la tarea, describe detalladamente qué pruebas has realizado e indica
   la GPU y controlador si ejecutaste pruebas gráficas. Especifica si utilizaste asistencia de IA conforme a
   [`AGENTS.md`](../../../AGENTS.md). Las normas completas se encuentran en [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

La [guía de pruebas de GPU](../../testing.md) explica cómo ejecutar las pruebas sin necesidad de archivos del juego.
Publica los resultados en la [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166) (un comentario por GPU y
controlador).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
