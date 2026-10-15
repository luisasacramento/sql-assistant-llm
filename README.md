# Text-to-SQL com LLM local (Qwen2.5 + QLoRA)

Projeto de fine-tuning de uma LLM local, 100% em GPU, para converter perguntas em linguagem natural em consultas SQL, com frontend em Gradio.

**Grupo:** (nomes dos integrantes)

## 1. Problema
Dado o schema de uma tabela (`CREATE TABLE ...`) e uma pergunta, o modelo gera a query SQL. Tem avaliação objetiva e é útil na prática.

## 2. Modelo
`Qwen/Qwen2.5-1.5B-Instruct`: pequeno, licença aberta, cabe em ~6 GB de VRAM com quantização 4-bit (QLoRA). Fine-tuning com LoRA (só ~1% dos parâmetros é treinado).

## 3. Hardware
- GPU: **(modelo e VRAM)**
- Driver / CUDA / PyTorch: **(versões)**
- ![GPU no VSCode](docs/print_gpu_vscode.png)

## 4. Como executar
```bash
# 1. ambiente: veja docs/SETUP_GPU.md
python scripts/check_gpu.py
# 2. dados (5000 treino / 200 teste, sem sobreposição)
python data/prepare_data.py
# 3. um treino simples
python src/train.py --rank 16 --lr 2e-4 --epochs 1 --name E2_r16
python src/evaluate.py --name E2_r16 --adapter results/E2_r16/adapter
# 4. grade completa (baseline + 5 experimentos + temperatura) e gráficos
python src/experiments.py
# 5. frontend
python app/app.py
```

## 5. Dataset e prompt
`b-mc2/sql-create-context` (derivado de WikiSQL e Spider). Formato: `### Schema / ### Pergunta / ### SQL`. A loss é calculada apenas sobre a resposta.

## 6. Experimentos
| Exp | Rank | LR | Épocas | Loss final | Exact match | SQL válido | Tempo (s) | VRAM pico (GB) |
|---|---|---|---|---|---|---|---|---|
| Baseline | - | - | - | - | | | - | - |
| E1 | 8 | 2e-4 | 1 | | | | | |
| E2 | 16 | 2e-4 | 1 | | | | | |
| E3 | 32 | 2e-4 | 1 | | | | | |
| E4 | 16 | 1e-4 | 1 | | | | | |
| E5 | 16 | 2e-4 | 3 | | | | | |
| E2 + temp 0.7 | 16 | 2e-4 | 1 | | | | | |

(preencher com `results/summary.csv`)

![Comparação](docs/comparacao_experimentos.png)
![Loss](docs/curvas_loss.png)

## 7. Métricas
- **Exact match** normalizado (minúsculas, espaços e aspas normalizados).
- **SQL válido**: a query é aceita pelo SQLite (`EXPLAIN`) usando o schema do exemplo.
- Limitação: exact match penaliza queries equivalentes escritas de outra forma, e a validade não garante resposta correta (não há dados nas tabelas).

## 8. Análise qualitativa
(Inserir 10 exemplos de `results/eval_*.json`: pergunta, base, fine-tuned, esperado; e os erros mais comuns.)

## 9. Frontend
(Print de `app/app.py` aqui.) Permite digitar schema e pergunta, escolher o experimento, comparar com o modelo base e ajustar temperatura e tamanho máximo.

## 10. Limitações e trabalhos futuros
Dataset em inglês, schemas de uma tabela, treino com 5000 exemplos. Futuro: mais dados, Spider completo, avaliação por execução com dados reais.

## 11. Divisão de tarefas
| Integrante | Responsabilidade |
|---|---|
| | Ambiente/GPU/CUDA |
| | Dados |
| | Treino/experimentos |
| | Avaliação |
| | Frontend/documentação |

## 12. Referências
Hu et al., LoRA (2021); Dettmers et al., QLoRA (2023); Hugging Face transformers/peft; Qwen2.5; b-mc2/sql-create-context.
