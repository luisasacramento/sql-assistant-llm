# Sem GPU NVIDIA local: usando a GPU do Google Colab dentro do VSCode

1. Instale no VSCode as extensões **Jupyter** e **Colab** (a oficial, publicada pelo Google).
2. Abra `notebooks/colab_run.ipynb`.
3. Em "Select Kernel" escolha **Colab > New Colab Server**, entre com a conta Google e selecione GPU **T4**.
4. Execute as células em ordem. A primeira roda `!nvidia-smi` e o `check_gpu.py`: **esse é o print exigido** (VSCode + GPU funcionando).

Observações
- O runtime do Colab é remoto: ele não enxerga arquivos do seu PC. Por isso o notebook faz `git clone` do repositório do GitHub.
- A extensão GPU Monitor monitora a GPU local e não funciona aqui. Use `!nvidia-smi` (ou `!nvidia-smi -l 5` em outra célula) durante o treino.
- A GPU gratuita (T4, ~15 GB) não é garantida em horários de pico, e a sessão cai por inatividade. Salve os resultados no GitHub ao final (última célula do notebook).
- Avise o professor que a GPU é remota (Colab). O enunciado fala em "local".
