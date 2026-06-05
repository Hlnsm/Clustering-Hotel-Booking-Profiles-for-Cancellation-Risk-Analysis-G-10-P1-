# Projeto de Clustering de Reservas de Hotel

Este projeto corre a pipeline completa para o dataset de reservas de hotel:
preparacao dos dados, exploracao de clustering, registo das experiencias,
profiling dos clusters selecionados, tabelas de taxa de cancelamento e tempo de
execucao.

## Estrutura do Repositorio

- `run_all.py` - ponto de entrada principal da pipeline.
- `src/` - codigo fonte do projeto.
- `data/` - ficheiros de dados de entrada.
- `tables/` - tabelas geradas, incluindo `experiments.csv`.
- `figures/` - figuras geradas.
- `requirements-lock.txt` - versoes fixas das dependencias Python usadas para reproducibilidade.

## Setup

Criar e ativar uma virtual environment e instalar as dependencias:

```powershell
python -m venv ..\venv
..\venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
```

## Run Completa

Para correr a pipeline completa a partir da raiz do projeto:

```powershell
python .\run_all.py
```

Esta run regenera as figuras e tabelas reportadas. Pode demorar bastante tempo,
porque faz as exploracoes completas de k-means, iK-means e GMM no dataset total.

Outputs principais:

- `tables/experiments.csv`
- `tables/runtime_full_pipeline.csv`
- `tables/R0_Euclid_standard_noADR/`
- `tables/R1_Euclid_robust_noADR/`
- `tables/clusterProfiles/`
- `figures/dataPreparation/`

## Modo Fast Check

Para verificar rapidamente se a pipeline corre e se as tabelas mantem o schema
esperado:

```powershell
python .\run_all.py fast
```

O modo fast usa grelhas reduzidas:

- k-means: `K=2,3`, `M=2`
- iK-means: `min_cluster_size=3000,5000`
- GMM: `K=2,3`, `M=2`, covariance types `diag` e `tied`

Os outputs das exploracoes em modo fast sao guardados em:

- `tables/fast_check/`

O modo fast nao corre o profiling post-hoc das runs finais selecionadas, porque
esses perfis dependem das configuracoes finais da run completa.

Nota: as tabelas e figuras da preparacao dos dados continuam a ser regeneradas
nas pastas normais (`tables/dataPreparation/` e `figures/dataPreparation/`).

## Dados

O dataset esperado deve estar em:

```text
data/hotel_bookings_course_release_v1.csv
```

A pipeline verifica a pasta `data/` antes de correr.
