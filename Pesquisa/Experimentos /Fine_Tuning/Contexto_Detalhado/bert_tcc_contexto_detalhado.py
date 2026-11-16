# ============================================================
# TCC - Fine-tuning e comparação de modelos BERT
# Google Colab
#
# Experimentos:
# 1. BERTimbau sem contexto
# 2. BERTimbau com contexto
# 3. DistilBERT multilíngue sem contexto
# 4. DistilBERT multilíngue com contexto
#
# Arquivos de entrada esperados:
#   videos.csv: id_video e contexto (ou titulo/descricao)
#   comentarios.csv: id_video, comentario, positivo, negativo, neutro
#
# ============================================================

import os
import re
import json
import random
import shutil
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import emoji

from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_VIDEOS = "/content/videos.csv"
ARQUIVO_COMENTARIOS = "/content/comentarios.csv"

# Para executar todos os experimentos previstos na metodologia:
MODELOS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "distilbert": "distilbert/distilbert-base-multilingual-cased",
}

# Mude para False apenas se quiser testar o código rapidamente.
EXECUTAR_TODOS_EXPERIMENTOS = True

# True: divide por id_video, evitando que comentários do mesmo vídeo
# apareçam simultaneamente em treino, validação e teste.
# Com cerca de 13 vídeos, as proporções serão aproximadas (por exemplo,
# 9 vídeos para treino, 2 para validação e 2 para teste).
DIVISAO_POR_VIDEO = True

# A partição treino/validação/teste é criada uma vez e depois reutilizada.
SEED_DIVISAO = 43

# Três sementes para repetir o treinamento com a mesma partição.
SEMENTES_TREINO = [42, 123, 2026]

# Os dois modelos utilizados aceitam sequências de até 512 tokens.
# Comece com 256 se a GPU tiver pouca memória; use 512 para aproveitar
# contextos mais longos, monitorando o consumo de VRAM.
MAX_LENGTH = 512
TAXA_APRENDIZADO = 2e-5
BATCH_TREINO = 4
BATCH_AVALIACAO = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCAS = 4
WEIGHT_DECAY = 0.01

PASTA_RESULTADOS = Path("/content/resultados_tcc_contexto_512")
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

NOMES_CLASSES = ["NEGATIVO", "NEUTRO", "POSITIVO"]
ID2LABEL = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}
LABEL2ID = {"NEGATIVO": 0, "NEUTRO": 1, "POSITIVO": 2}
CLASSES = np.array([0, 1, 2])

# Padrões deliberadamente conservadores: revise-os conforme o corpus.
PADROES_SPAM = [
    r"\bganhe dinheiro\b",
    r"\brenda extra\b",
    r"\bcupom de desconto\b",
    r"\bcompre agora\b",
    r"\bclique aqui para ganhar\b",
    r"\bpromo[cç][aã]o imperd[ií]vel\b",
]

# Colunas opcionais que podem conter o comentário ao qual se responde.
COLUNAS_COMENTARIO_PAI = [
    "comentario_pai",
    "comentario_respondido",
    "texto_comentario_pai",
    "comentario_original",
]

# ============================================================
# 2. REPRODUTIBILIDADE E AMBIENTE
# ============================================================

random.seed(SEED_DIVISAO)
np.random.seed(SEED_DIVISAO)
torch.manual_seed(SEED_DIVISAO)
set_seed(SEED_DIVISAO)

print("=" * 72)
print("AMBIENTE")
print("=" * 72)
print("PyTorch:", torch.__version__)
print("GPU disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    warnings.warn(
        "GPU não detectada. O treinamento pode ser muito lento. "
        "No Colab, selecione Ambiente de execução > Alterar tipo > GPU."
    )

# ============================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================

def normalizar_texto(texto):
    """Converte emojis em descrições e normaliza ruídos textuais."""
    if pd.isna(texto):
        return ""

    texto = str(texto)

    # Preserva o significado do emoji por meio de uma descrição textual.
    texto = emoji.demojize(texto, language="pt")

    # Remoção de URLs e menções.
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"(?<!\w)@\w+", " ", texto)

    # Reduz sequências excessivas de pontuação, sem remover a pontuação normal.
    texto = re.sub(r"([!?.]){3,}", r"\1", texto)

    # Normaliza espaços. Não converte para minúsculas, pois o BERTimbau é cased.
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def somente_ruido(texto_original):
    """
    Retorna True quando, após retirar emojis, URLs, menções e pontuação,
    não resta conteúdo textual significativo.
    """
    if pd.isna(texto_original):
        return True

    texto = str(texto_original)
    texto = emoji.replace_emoji(texto, replace=" ")
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"(?<!\w)@\w+", " ", texto)
    texto = re.sub(r"[\W_]+", " ", texto, flags=re.UNICODE)
    return len(texto.split()) == 0


def parece_spam(texto):
    """Heurística conservadora para padrões promocionais explícitos."""
    texto_lower = str(texto).lower()
    return any(re.search(padrao, texto_lower) for padrao in PADROES_SPAM)


def encontrar_coluna_comentario_pai(df):
    for coluna in COLUNAS_COMENTARIO_PAI:
        if coluna in df.columns:
            return coluna
    return None


# ============================================================
# 4. CARREGAR E VALIDAR OS CSVs
# ============================================================

if not Path(ARQUIVO_VIDEOS).exists():
    raise FileNotFoundError(
        f"Não encontrei {ARQUIVO_VIDEOS}. Envie videos.csv para /content."
    )
if not Path(ARQUIVO_COMENTARIOS).exists():
    raise FileNotFoundError(
        f"Não encontrei {ARQUIVO_COMENTARIOS}. Envie comentarios.csv para /content."
    )

videos = pd.read_csv(ARQUIVO_VIDEOS, encoding="utf-8-sig")
comentarios = pd.read_csv(ARQUIVO_COMENTARIOS, encoding="utf-8-sig")

if "id_video" not in videos.columns:
    raise ValueError("videos.csv precisa ter a coluna 'id_video'.")
if "id_video" not in comentarios.columns or "comentario" not in comentarios.columns:
    raise ValueError("comentarios.csv precisa ter 'id_video' e 'comentario'.")

colunas_rotulos = ["positivo", "negativo", "neutro"]
faltantes = [c for c in colunas_rotulos if c not in comentarios.columns]
if faltantes:
    raise ValueError(f"Faltam colunas de rótulos em comentarios.csv: {faltantes}")

# Aceita a coluna contexto já existente ou combina título/descrição.
if "contexto" not in videos.columns:
    campos_contexto = [c for c in ["titulo", "descricao", "descrição"] if c in videos.columns]
    if not campos_contexto:
        raise ValueError(
            "videos.csv deve conter 'contexto' ou pelo menos uma coluna "
            "entre 'titulo', 'descricao' e 'descrição'."
        )
    videos["contexto"] = videos[campos_contexto].fillna("").astype(str).agg(" ".join, axis=1)
else:
    videos["contexto"] = videos["contexto"].fillna("").astype(str)

# Se título/descrição existirem além de contexto, acrescenta-os sem duplicar
# automaticamente o campo contexto.
campos_adicionais = [c for c in ["titulo", "descricao", "descrição"] if c in videos.columns]
if campos_adicionais:
    adicionais = videos[campos_adicionais].fillna("").astype(str).agg(" ".join, axis=1)
    videos["contexto"] = (videos["contexto"].fillna("") + " " + adicionais).str.strip()

videos["id_video"] = videos["id_video"].astype(str).str.strip()
comentarios["id_video"] = comentarios["id_video"].astype(str).str.strip()

# Evita multiplicar comentários se videos.csv tiver IDs duplicados.
if videos["id_video"].duplicated().any():
    duplicados = videos.loc[videos["id_video"].duplicated(), "id_video"].tolist()
    raise ValueError(f"Há id_video duplicado em videos.csv: {duplicados[:10]}")

print("\nVídeos carregados:", len(videos))
print("Comentários brutos:", len(comentarios))

df = comentarios.merge(
    videos[["id_video", "contexto"]],
    on="id_video",
    how="left",
    validate="many_to_one",
)

if df["contexto"].isna().any():
    ids_sem_contexto = df.loc[df["contexto"].isna(), "id_video"].unique().tolist()
    raise ValueError(
        "Existem comentários sem correspondência em videos.csv. "
        f"IDs (até 10): {ids_sem_contexto[:10]}"
    )

coluna_pai = encontrar_coluna_comentario_pai(df)
if coluna_pai:
    print("Coluna de comentário respondido detectada:", coluna_pai)
else:
    print(
        "Nenhuma coluna de comentário respondido foi detectada. "
        "O contexto usará as informações disponíveis em videos.csv."
    )

# ============================================================
# 5. LIMPEZA, FILTRAGEM E RÓTULOS
# ============================================================

df["comentario_original"] = df["comentario"].fillna("").astype(str)
df["contexto_original"] = df["contexto"].fillna("").astype(str)

# Converte rótulos em valores numéricos e valida se cada linha tem exatamente um 1.
for coluna in colunas_rotulos:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0).astype(int)

valores_validos = df[colunas_rotulos].isin([0, 1]).all(axis=1)
soma_rotulos = df[colunas_rotulos].sum(axis=1)
rotulos_invalidos = (~valores_validos) | (soma_rotulos != 1)

if rotulos_invalidos.any():
    exemplos = df.loc[
        rotulos_invalidos, ["comentario", *colunas_rotulos]
    ].head(10)
    raise ValueError(
        "Há rótulos inválidos. Cada comentário precisa ter exatamente uma "
        f"classe igual a 1 e as demais iguais a 0.\n{exemplos}"
    )

def criar_label(linha):
    if linha["negativo"] == 1:
        return 0
    if linha["neutro"] == 1:
        return 1
    return 2

df["label"] = df.apply(criar_label, axis=1)

contagem_filtragem = {}
contagem_filtragem["brutos"] = len(df)

# Descarta comentários vazios ou compostos apenas por emojis/URLs/menções.
mask_ruido = df["comentario_original"].apply(somente_ruido)
contagem_filtragem["removidos_vazios_ou_somente_emojis_urls_mencoes"] = int(mask_ruido.sum())
df = df.loc[~mask_ruido].copy()

# Normaliza comentários e contexto.
df["comentario"] = df["comentario_original"].apply(normalizar_texto)
df["contexto"] = df["contexto_original"].apply(normalizar_texto)

if coluna_pai:
    df["comentario_pai_processado"] = df[coluna_pai].fillna("").astype(str).apply(normalizar_texto)
else:
    df["comentario_pai_processado"] = ""

# Remove texto vazio após normalização.
mask_vazio = df["comentario"].str.strip().eq("")
contagem_filtragem["removidos_apos_normalizacao"] = int(mask_vazio.sum())
df = df.loc[~mask_vazio].copy()

# Não removemos comentários duplicados nem comentários curtos:
# o usuário já realizou essa curadoria, e textos iguais podem ter sentidos
# diferentes conforme o contexto do vídeo ou da conversa.
contagem_filtragem["duplicatas_removidas"] = 0
contagem_filtragem["comentarios_curtos_removidos"] = 0

# Remove spam de padrões explícitos e conservadores.
mask_spam = df["comentario"].apply(parece_spam)
contagem_filtragem["removidos_spam_heuristico"] = int(mask_spam.sum())
df = df.loc[~mask_spam].copy()

contagem_filtragem["final"] = len(df)

print("\nResumo da filtragem:")
for etapa, quantidade in contagem_filtragem.items():
    print(f"  {etapa}: {quantidade}")

if df.empty:
    raise ValueError("Nenhum comentário restou após a filtragem. Revise os critérios.")

# Entrada com contexto: usa o contexto do vídeo e, se houver, o comentário respondido.
def montar_texto_com_contexto(linha):
    partes = []
    if linha["contexto"].strip():
        partes.append("CONTEXTO DO VIDEO: " + linha["contexto"])
    if linha["comentario_pai_processado"].strip():
        partes.append("COMENTARIO AO QUAL RESPONDE: " + linha["comentario_pai_processado"])
    partes.append("COMENTARIO: " + linha["comentario"])
    return " [SEP] ".join(partes)

df["texto_com_contexto"] = df.apply(montar_texto_com_contexto, axis=1)
df["texto_sem_contexto"] = "COMENTARIO: " + df["comentario"]

print("\nDistribuição das classes após filtragem:")
print(df["label"].map(ID2LABEL).value_counts().reindex(NOMES_CLASSES, fill_value=0))

# ============================================================
# 6. DIVISÃO FIXA TREINO / VALIDAÇÃO / TESTE
# ============================================================

def dividir_por_video(dados, seed=SEED_DIVISAO, tentativas=500):
    """
    Procura uma divisão por id_video próxima de 70/15/15 e que mantenha
    as três classes em cada subconjunto. Todos os comentários de um vídeo
    ficam no mesmo subconjunto.
    """
    grupos = dados["id_video"].astype(str).to_numpy()
    y = dados["label"].to_numpy()
    indices = np.arange(len(dados))
    melhor = None
    melhor_pontuacao = float("inf")
    encontrou_todas_classes = False

    for tentativa in range(tentativas):
        s = seed + tentativa

        primeiro = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=s)
        idx_train, idx_temp = next(primeiro.split(indices, y, groups=grupos))

        temp = dados.iloc[idx_temp]
        grupos_temp = temp["id_video"].astype(str).to_numpy()
        idx_temp_local = np.arange(len(temp))

        segundo = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=s + 10000)
        idx_val_local, idx_test_local = next(
            segundo.split(idx_temp_local, temp["label"].to_numpy(), groups=grupos_temp)
        )

        idx_val = idx_temp[idx_val_local]
        idx_test = idx_temp[idx_test_local]

        partes = [
            dados.iloc[idx_train],
            dados.iloc[idx_val],
            dados.iloc[idx_test],
        ]
        proporcoes = np.array([len(p) for p in partes]) / len(dados)
        alvo = np.array([0.70, 0.15, 0.15])
        pontuacao = float(np.abs(proporcoes - alvo).sum())

        classes_completas = all(set(p["label"].unique()) == set(CLASSES) for p in partes)
        # Prefere divisões com todas as classes e proporções próximas.
        if classes_completas:
            pontuacao -= 10
        if pontuacao < melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor = (idx_train, idx_val, idx_test, classes_completas, s)
        if classes_completas and np.abs(proporcoes - alvo).sum() < 0.08:
            encontrou_todas_classes = True
            break

    if melhor is None:
        raise RuntimeError("Não foi possível construir uma divisão por vídeo.")

    idx_train, idx_val, idx_test, completas, seed_escolhida = melhor
    if not completas:
        warnings.warn(
            "Não foi encontrada uma divisão por vídeo que contenha as três classes "
            "em todos os subconjuntos. Confira as classes por vídeo e considere "
            "revisar os dados ou documentar uma estratégia alternativa."
        )

    return (
        dados.iloc[idx_train].copy(),
        dados.iloc[idx_val].copy(),
        dados.iloc[idx_test].copy(),
        seed_escolhida,
    )


def dividir_estratificado_por_comentario(dados):
    """Alternativa que estratifica por classe, mas não impede vazamento por vídeo."""
    treino, temporario = train_test_split(
        dados, test_size=0.30, random_state=SEED_DIVISAO, stratify=dados["label"]
    )
    validacao, teste = train_test_split(
        temporario, test_size=0.50, random_state=SEED_DIVISAO, stratify=temporario["label"]
    )
    return treino.copy(), validacao.copy(), teste.copy(), SEED_DIVISAO


# Arquivos persistentes da partição. Se já existirem, serão reutilizados
# em todas as sementes e em execuções futuras do script.
ARQUIVO_TREINO = PASTA_RESULTADOS / "divisao_treino.csv"
ARQUIVO_VALIDACAO = PASTA_RESULTADOS / "divisao_validacao.csv"
ARQUIVO_TESTE = PASTA_RESULTADOS / "divisao_teste.csv"
ARQUIVOS_PARTICAO = [ARQUIVO_TREINO, ARQUIVO_VALIDACAO, ARQUIVO_TESTE]

if all(caminho.exists() for caminho in ARQUIVOS_PARTICAO):
    print("\nPartição salva encontrada. Reutilizando os CSVs existentes.")
    train_df = pd.read_csv(ARQUIVO_TREINO, encoding="utf-8-sig")
    validation_df = pd.read_csv(ARQUIVO_VALIDACAO, encoding="utf-8-sig")
    test_df = pd.read_csv(ARQUIVO_TESTE, encoding="utf-8-sig")

    colunas_necessarias = set(df.columns)
    for nome_parte, parte in [
        ("treino", train_df), ("validação", validation_df), ("teste", test_df)
    ]:
        faltantes = colunas_necessarias - set(parte.columns)
        if faltantes:
            raise ValueError(
                f"A partição salva de {nome_parte} não contém as colunas esperadas: "
                f"{sorted(faltantes)}. Apague os três CSVs de partição para recriá-los."
            )
    metodo_divisao = "partição salva reutilizada"
    seed_divisao = SEED_DIVISAO
else:
    print("\nCriando a partição do dataset pela primeira vez.")
    if DIVISAO_POR_VIDEO:
        train_df, validation_df, test_df, seed_divisao = dividir_por_video(
            df, seed=SEED_DIVISAO
        )
        metodo_divisao = "group-aware por id_video"
    else:
        train_df, validation_df, test_df, seed_divisao = dividir_estratificado_por_comentario(df)
        metodo_divisao = (
            "estratificada por comentário (pode haver comentários do mesmo vídeo "
            "em conjuntos diferentes)"
        )

    train_df.assign(subconjunto="treino").to_csv(
        ARQUIVO_TREINO, index=False, encoding="utf-8-sig"
    )
    validation_df.assign(subconjunto="validacao").to_csv(
        ARQUIVO_VALIDACAO, index=False, encoding="utf-8-sig"
    )
    test_df.assign(subconjunto="teste").to_csv(
        ARQUIVO_TESTE, index=False, encoding="utf-8-sig"
    )

def relatorio_divisao(nome, parte):
    total_particionado = len(train_df) + len(validation_df) + len(test_df)
    print(f"\n{nome}: {len(parte)} comentários ({len(parte)/total_particionado:.1%})")
    print("  vídeos:", parte["id_video"].nunique())
    print("  classes:")
    print(parte["label"].map(ID2LABEL).value_counts().reindex(NOMES_CLASSES, fill_value=0).to_dict())

print("\nMétodo de divisão:", metodo_divisao)
print("Seed usada para divisão:", seed_divisao)
relatorio_divisao("TREINO", train_df)
relatorio_divisao("VALIDAÇÃO", validation_df)
relatorio_divisao("TESTE", test_df)

if DIVISAO_POR_VIDEO:
    grupos_treino = set(train_df["id_video"].astype(str))
    grupos_val = set(validation_df["id_video"].astype(str))
    grupos_teste = set(test_df["id_video"].astype(str))
    assert grupos_treino.isdisjoint(grupos_val), "Há vídeos compartilhados entre treino e validação."
    assert grupos_treino.isdisjoint(grupos_teste), "Há vídeos compartilhados entre treino e teste."
    assert grupos_val.isdisjoint(grupos_teste), "Há vídeos compartilhados entre validação e teste."

with open(PASTA_RESULTADOS / "configuracao.json", "w", encoding="utf-8") as f:
    json.dump({
        "semente_divisao": int(seed_divisao),
        "sementes_treino": SEMENTES_TREINO,
        "metodo_divisao": metodo_divisao,
        "divisao_por_video": DIVISAO_POR_VIDEO,
        "proporcoes_observadas": {
            "treino": len(train_df) / (len(train_df) + len(validation_df) + len(test_df)),
            "validacao": len(validation_df) / (len(train_df) + len(validation_df) + len(test_df)),
            "teste": len(test_df) / (len(train_df) + len(validation_df) + len(test_df)),
        },
        "contagem_filtragem": contagem_filtragem,
        "max_length": MAX_LENGTH,
        "learning_rate": TAXA_APRENDIZADO,
        "batch_train": BATCH_TREINO,
        "batch_eval": BATCH_AVALIACAO,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "padding": "dinamico_com_DataCollatorWithPadding",
        "epochs": EPOCAS,
        "weighted_cross_entropy": True,
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# 7. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ============================================================

class WeightedTrainer(Trainer):
    """Trainer com Cross-Entropy ponderada por classe."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("É necessário fornecer class_weights.")
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def calcular_metricas(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def preparar_dataset(parte, coluna_texto, tokenizer):
    dataset = Dataset.from_pandas(
        parte[[coluna_texto, "label"]].rename(columns={coluna_texto: "texto"}),
        preserve_index=False,
    )

    def tokenizar(exemplos):
        return tokenizer(
            exemplos["texto"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenizar, batched=True)
    return dataset


def executar_experimento(nome_modelo, id_modelo, usar_contexto, seed_treino):
    condicao = "com_contexto" if usar_contexto else "sem_contexto"
    nome_experimento = f"{nome_modelo}_{condicao}"
    pasta_exp = PASTA_RESULTADOS / f"semente_{seed_treino}" / nome_experimento
    pasta_exp.mkdir(parents=True, exist_ok=True)
    pasta_checkpoints = pasta_exp / "checkpoints"
    pasta_modelo_final = pasta_exp / "modelo_final"

    coluna_texto = "texto_com_contexto" if usar_contexto else "texto_sem_contexto"

    print("\n" + "=" * 78)
    print("EXPERIMENTO:", nome_experimento)
    print("Semente de treinamento:", seed_treino)
    print("Modelo inicial:", id_modelo)
    print("Entrada:", "contexto + comentário" if usar_contexto else "somente comentário")
    print("=" * 78)

    # Reinicializa os geradores para que cada execução use sua semente.
    random.seed(seed_treino)
    np.random.seed(seed_treino)
    torch.manual_seed(seed_treino)
    set_seed(seed_treino)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_treino)

    # Cada experimento carrega o checkpoint pré-treinado original.
    tokenizer = AutoTokenizer.from_pretrained(id_modelo)

    # Confere o limite arquitetural do modelo antes de tokenizar.
    model_config = AutoConfig.from_pretrained(id_modelo)
    limite_modelo = getattr(model_config, "max_position_embeddings", MAX_LENGTH)
    if MAX_LENGTH > limite_modelo:
        raise ValueError(
            f"MAX_LENGTH={MAX_LENGTH} excede o limite do modelo ({limite_modelo})."
        )
    del model_config
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ds_treino = preparar_dataset(train_df, coluna_texto, tokenizer)
    ds_validacao = preparar_dataset(validation_df, coluna_texto, tokenizer)
    ds_teste = preparar_dataset(test_df, coluna_texto, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        id_modelo,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    contagem = train_df["label"].value_counts().reindex(CLASSES, fill_value=0)
    if (contagem == 0).any():
        raise ValueError(
            f"Uma classe não aparece no treino para {nome_experimento}: {contagem.to_dict()}"
        )

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=CLASSES,
        y=train_df["label"].to_numpy(),
    )
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

    print("Contagem no treino:", {
        ID2LABEL[int(k)]: int(v) for k, v in contagem.items()
    })
    print("Pesos Weighted Cross-Entropy:", {
        ID2LABEL[i]: round(float(pesos[i]), 4) for i in range(3)
    })

    # Padding dinâmico: cada lote é preenchido apenas até o maior exemplo
    # daquele lote, reduzindo memória em comparação com padding fixo de 512.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    argumentos = TrainingArguments(
        output_dir=str(pasta_checkpoints),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=TAXA_APRENDIZADO,
        per_device_train_batch_size=BATCH_TREINO,
        per_device_eval_batch_size=BATCH_AVALIACAO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCAS,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=seed_treino,
        data_seed=seed_treino,
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=argumentos,
        train_dataset=ds_treino,
        eval_dataset=ds_validacao,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=calcular_metricas,
        class_weights=pesos_tensor,
    )

    trainer.train()

    # Salva o melhor checkpoint selecionado pela macro-F1 de validação.
    trainer.save_model(str(pasta_modelo_final))
    tokenizer.save_pretrained(str(pasta_modelo_final))

    # Remove checkpoints intermediários para economizar o armazenamento do Colab.
    # O melhor modelo já foi carregado por load_best_model_at_end e foi salvo acima.
    shutil.rmtree(pasta_checkpoints, ignore_errors=True)

    # Avaliação final no teste, sem ajustar parâmetros com base nele.
    resultado_teste = trainer.evaluate(ds_teste)
    predicoes = trainer.predict(ds_teste)
    y_true = predicoes.label_ids
    y_pred = np.argmax(predicoes.predictions, axis=-1)

    relatorio_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=NOMES_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    relatorio_texto = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=NOMES_CLASSES,
        digits=4,
        zero_division=0,
    )
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Salva relatório e matriz de confusão.
    with open(pasta_exp / "relatorio_classificacao.txt", "w", encoding="utf-8") as f:
        f.write(relatorio_texto)

    pd.DataFrame(
        matriz,
        index=[f"real_{n}" for n in NOMES_CLASSES],
        columns=[f"previsto_{n}" for n in NOMES_CLASSES],
    ).to_csv(pasta_exp / "matriz_confusao.csv", encoding="utf-8-sig")

    # Guarda as previsões individuais para análise qualitativa dos erros.
    previsoes_df = test_df[[
        "id_video", "comentario_original", "comentario",
        "contexto_original", "label"
    ]].copy()
    previsoes_df["rotulo_real"] = previsoes_df["label"].map(ID2LABEL)
    previsoes_df["label_previsto"] = y_pred
    previsoes_df["rotulo_previsto"] = previsoes_df["label_previsto"].map(ID2LABEL)
    previsoes_df["correto"] = previsoes_df["label"] == previsoes_df["label_previsto"]
    previsoes_df["texto_usado"] = test_df[coluna_texto].to_numpy()
    previsoes_df.to_csv(
        pasta_exp / "predicoes_teste.csv", index=False, encoding="utf-8-sig"
    )

    # Guarda métricas e metadados.
    metricas_resumo = {
        "experimento": nome_experimento,
        "semente_treino": seed_treino,
        "modelo_pre_treinado": id_modelo,
        "condicao": condicao,
        "quantidade_treino": len(train_df),
        "quantidade_validacao": len(validation_df),
        "quantidade_teste": len(test_df),
        "contagem_classes_treino": {
            ID2LABEL[int(k)]: int(v) for k, v in contagem.items()
        },
        "pesos_classes": {
            ID2LABEL[i]: float(pesos[i]) for i in range(3)
        },
        "metricas_teste_trainer": {
            k: float(v) for k, v in resultado_teste.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        },
        "metricas_classification_report": relatorio_dict,
        "matriz_confusao": matriz.tolist(),
    }
    with open(pasta_exp / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas_resumo, f, ensure_ascii=False, indent=2)

    print("\nRelatório de classificação:")
    print(relatorio_texto)
    print("Matriz de confusão:\n", matriz)
    print("Resultados salvos em:", pasta_exp)

    # Libera memória antes do próximo modelo.
    del trainer, model, tokenizer, ds_treino, ds_validacao, ds_teste
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experimento": nome_experimento,
        "semente_treino": seed_treino,
        "modelo": id_modelo,
        "condicao": condicao,
        "accuracy": float(relatorio_dict["accuracy"]),
        "f1_macro": float(relatorio_dict["macro avg"]["f1-score"]),
        "f1_negativo": float(relatorio_dict["NEGATIVO"]["f1-score"]),
        "f1_neutro": float(relatorio_dict["NEUTRO"]["f1-score"]),
        "f1_positivo": float(relatorio_dict["POSITIVO"]["f1-score"]),
        "precision_macro": float(relatorio_dict["macro avg"]["precision"]),
        "recall_macro": float(relatorio_dict["macro avg"]["recall"]),
    }


# ============================================================
# 8. EXECUTAR OS EXPERIMENTOS
# ============================================================

if EXECUTAR_TODOS_EXPERIMENTOS:
    modelos_a_executar = MODELOS
else:
    modelos_a_executar = {"bertimbau": MODELOS["bertimbau"]}

resumos = []
for seed_treino in SEMENTES_TREINO:
    for nome_modelo, id_modelo in modelos_a_executar.items():
        for usar_contexto in [False, True]:
            resumo = executar_experimento(
                nome_modelo, id_modelo, usar_contexto, seed_treino
            )
            resumos.append(resumo)

comparacao = pd.DataFrame(resumos)
comparacao = comparacao.sort_values(
    by=["modelo", "condicao", "semente_treino"]
).reset_index(drop=True)
comparacao.to_csv(
    PASTA_RESULTADOS / "resultados_todas_execucoes.csv",
    index=False,
    encoding="utf-8-sig",
)

metricas = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
agrupado = comparacao.groupby(["modelo", "condicao"])[metricas].agg(["mean", "std"])
agrupado.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in agrupado.columns]
agrupado = agrupado.reset_index().rename(columns={
    "accuracy_mean": "accuracy_media",
    "accuracy_std": "accuracy_desvio_padrao",
    "precision_macro_mean": "precision_macro_media",
    "precision_macro_std": "precision_macro_desvio_padrao",
    "recall_macro_mean": "recall_macro_media",
    "recall_macro_std": "recall_macro_desvio_padrao",
    "f1_macro_mean": "f1_macro_media",
    "f1_macro_std": "f1_macro_desvio_padrao",
})
agrupado.to_csv(
    PASTA_RESULTADOS / "media_desvio_padrao.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 78)
print("RESULTADOS INDIVIDUAIS DAS EXECUÇÕES")
print("=" * 78)
print(comparacao.to_string(index=False))
print("\n" + "=" * 78)
print("MÉDIA E DESVIO-PADRÃO ENTRE AS SEMENTES")
print("=" * 78)
print(agrupado.to_string(index=False))

# Compacta resultados para facilitar o download ou cópia para o Google Drive.
arquivo_zip = "/content/resultados_tcc.zip"
if os.path.exists(arquivo_zip):
    os.remove(arquivo_zip)

with zipfile.ZipFile(arquivo_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for arquivo in PASTA_RESULTADOS.rglob("*"):
        if arquivo.is_file():
            zipf.write(arquivo, arcname=str(arquivo.relative_to(PASTA_RESULTADOS.parent)))

print("\nTreinamento concluído.")
print("Pasta de resultados:", PASTA_RESULTADOS)
print("Arquivo compactado:", arquivo_zip)
print(
    "Para baixar no Colab, execute em outra célula:\n"
    "from google.colab import files\n"
    "files.download('/content/resultados_tcc.zip')"
)
