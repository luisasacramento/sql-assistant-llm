# Configuração da GPU (Windows + VSCode)

## Erro: `nvidia-smi` não é reconhecido
Significa que o driver NVIDIA não está instalado ou a máquina não tem placa NVIDIA.

**Passo 1 - Descobrir qual placa existe** (PowerShell):
```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```
- Apareceu "NVIDIA GeForce/RTX/GTX ...": siga o passo 2.
- Só apareceu Intel/AMD: essa máquina **não roda CUDA**. Use outra máquina do grupo que tenha NVIDIA.

**Passo 2 - Instalar o driver:** baixe em https://www.nvidia.com/Download/index.aspx (escolha seu modelo) ou pelo app NVIDIA. Reinicie o PC e **abra um novo terminal**.

**Passo 3 - Testar:** `nvidia-smi`. Se ainda falhar, tente o caminho completo:
```powershell
& "C:\Windows\System32\nvidia-smi.exe"
& "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
```
Se funcionar só pelo caminho completo, adicione a pasta ao PATH.

**Notebook com 2 GPUs (Intel + NVIDIA):** o driver NVIDIA precisa estar instalado do mesmo jeito; use o Painel de Controle NVIDIA para a GPU de alto desempenho.

## Escolher o CUDA
`nvidia-smi` mostra no canto superior direito "CUDA Version: X.Y" = **máximo** suportado pelo driver. Instale um CUDA Toolkit igual ou menor (12.1 ou 12.4 são boas escolhas): https://developer.nvidia.com/cuda-toolkit-archive

Os pacotes do PyTorch já trazem o runtime CUDA, mas o trabalho pede o toolkit; confira com `nvcc --version`.

## Ambiente Python
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# PyTorch com CUDA (troque cu121 por cu124 se instalou o CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_gpu.py
```
Se o PowerShell bloquear o Activate: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Print obrigatório
No VSCode, deixe visíveis ao mesmo tempo: terminal com `nvidia-smi`, saída de `python scripts/check_gpu.py` e a extensão **GPU Monitor** (idealmente durante o treino). Salve em `docs/print_gpu_vscode.png`.

## Se faltar VRAM (CUDA out of memory)
Use `--bs 1`, `--max_len 256` ou o modelo `--model Qwen/Qwen2.5-0.5B-Instruct`.
