# ============================================================
# TCC - Balanceamento de classes: BERTimbau x DistilBERT
# Entrada: somente comentarios, sem contexto
#
# Estrategias:
#   original                  
#   weighted_cross_entropy   -> CrossEntropyLoss ponderada
#   oversampling             -> WeightedRandomSampler + CrossEntropy comum
#
# Usa comentarios.csv. A coluna id_video DEVE ser mantida para a divisao
# por video. Nao e necessario fornecer videos.csv.
# ============================================================

import os, re, json, random, shutil, zipfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import emoji

from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)

# ------------------------- CONFIGURACOES -------------------------

ARQUIVO_COMENTARIOS = "/content/comentarios.csv"

MODELOS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "distilbert": "distilbert/distilbert-base-multilingual-cased",
}
EXECUTAR_DOIS_MODELOS = True

ESTRATEGIAS = [
    "original",
    "weighted_cross_entropy",
    "oversampling",
]

SEMENTES_TREINO = [42, 123, 2026]
SEED_DIVISAO = 43
MAX_LENGTH = 512
TAXA_APRENDIZADO = 2e-5
BATCH_TREINO = 4
BATCH_AVALIACAO = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCAS = 4
WEIGHT_DECAY = 0.01

PASTA_RESULTADOS = Path("/content/resultados_balanceamento")
# Pasta do experimento anterior. Coloque aqui os tres CSVs de particao
# usados no experimento por video, caso queira reproduzir exatamente a divisao.
PASTA_PARTICOES_ANTERIORES = Path("/content/resultados_tcc_contexto_512")

NOMES_CLASSES = ["NEGATIVO", "NEUTRO", "POSITIVO"]
ID2LABEL = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}
LABEL2ID = {"NEGATIVO": 0, "NEUTRO": 1, "POSITIVO": 2}
CLASSES = np.array([0, 1, 2])

PADROES_SPAM = [
    r"\bganhe dinheiro\b", r"\brenda extra\b",
    r"\bcupom de desconto\b", r"\bcompre agora\b",
    r"\bclique aqui para ganhar\b", r"\bpromo[cç][aã]o imperd[ií]vel\b",
]

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# ------------------------- REPRODUTIBILIDADE -------------------------

random.seed(SEED_DIVISAO)
np.random.seed(SEED_DIVISAO)
torch.manual_seed(SEED_DIVISAO)
set_seed(SEED_DIVISAO)

print("PyTorch:", torch.__version__)
print("GPU disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    warnings.warn("GPU nao detectada. Se estiver no Colab, habilite uma GPU.")

# ------------------------- PRE-PROCESSAMENTO -------------------------

def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = emoji.demojize(str(texto), language="pt")
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto, flags=re.I)
    texto = re.sub(r"(?<!\w)@\w+", " ", texto)
    texto = re.sub(r"([!?.]){3,}", r"\1", texto)
    return re.sub(r"\s+", " ", texto).strip()


def somente_ruido(texto):
    if pd.isna(texto):
        return True
    texto = emoji.replace_emoji(str(texto), replace=" ")
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto, flags=re.I)
    texto = re.sub(r"(?<!\w)@\w+", " ", texto)
    texto = re.sub(r"[\W_]+", " ", texto, flags=re.UNICODE)
    return len(texto.split()) == 0


def parece_spam(texto):
    t = str(texto).lower()
    return any(re.search(p, t) for p in PADROES_SPAM)


# ------------------------- LEITURA DO DATASET -------------------------

if not Path(ARQUIVO_COMENTARIOS).exists():
    raise FileNotFoundError("Envie comentarios.csv para /content.")

df = pd.read_csv(ARQUIVO_COMENTARIOS, encoding="utf-8-sig")
obrigatorias = ["id_video", "comentario", "positivo", "negativo", "neutro"]
faltantes = [c for c in obrigatorias if c not in df.columns]
if faltantes:
    raise ValueError(f"Colunas ausentes em comentarios.csv: {faltantes}")

# NAO remova id_video: ele e usado para impedir que comentarios do mesmo
# video aparecam em treino, validacao e teste.
df["id_video"] = df["id_video"].astype(str).str.strip()
if df["id_video"].eq("").any():
    raise ValueError("Ha linhas sem id_video.")

df["comentario_original"] = df["comentario"].fillna("").astype(str)

for c in ["positivo", "negativo", "neutro"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

validos = df[["positivo", "negativo", "neutro"]].isin([0, 1]).all(axis=1)
soma = df[["positivo", "negativo", "neutro"]].sum(axis=1)
invalidos = (~validos) | (soma != 1)
if invalidos.any():
    raise ValueError(
        "Cada comentario deve ter exatamente um rotulo igual a 1. "
        f"Exemplos invalidos:\n{df.loc[invalidos, obrigatorias].head()}"
    )


def criar_label(linha):
    if linha["negativo"] == 1:
        return 0
    if linha["neutro"] == 1:
        return 1
    return 2


df["label"] = df.apply(criar_label, axis=1)
filtragem = {"brutos": len(df)}

ruido = df["comentario_original"].apply(somente_ruido)
filtragem["removidos_ruido"] = int(ruido.sum())
df = df.loc[~ruido].copy()

df["comentario"] = df["comentario_original"].apply(normalizar_texto)
vazios = df["comentario"].str.strip().eq("")
filtragem["removidos_apos_normalizacao"] = int(vazios.sum())
df = df.loc[~vazios].copy()

# Mantem duplicatas e comentarios curtos, como no script anterior.
filtragem["duplicatas_removidas"] = 0
filtragem["comentarios_curtos_removidos"] = 0

spam = df["comentario"].apply(parece_spam)
filtragem["removidos_spam_heuristico"] = int(spam.sum())
df = df.loc[~spam].copy()
filtragem["final"] = len(df)

if df.empty:
    raise ValueError("Dataset vazio apos filtragem.")

# Sem contexto: o modelo recebe somente o texto do comentario.
df["texto"] = "COMENTARIO: " + df["comentario"]

print("\nFiltragem:", filtragem)
print("\nDistribuicao geral:")
print(df["label"].map(ID2LABEL).value_counts().reindex(NOMES_CLASSES, fill_value=0))


# ------------------------- DIVISAO POR VIDEO -------------------------

def dividir_por_video(dados, seed=SEED_DIVISAO, tentativas=500):
    grupos = dados["id_video"].astype(str).to_numpy()
    y = dados["label"].to_numpy()
    indices = np.arange(len(dados))
    melhor = None
    melhor_score = float("inf")

    for tentativa in range(tentativas):
        s = seed + tentativa
        split1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=s)
        idx_train, idx_temp = next(split1.split(indices, y, groups=grupos))

        temp = dados.iloc[idx_temp]
        split2 = GroupShuffleSplit(
            n_splits=1, train_size=0.50, random_state=s + 10000
        )
        idx_val_local, idx_test_local = next(
            split2.split(
                np.arange(len(temp)),
                temp["label"].to_numpy(),
                groups=temp["id_video"].astype(str).to_numpy(),
            )
        )
        idx_val, idx_test = idx_temp[idx_val_local], idx_temp[idx_test_local]
        partes = [dados.iloc[idx_train], dados.iloc[idx_val], dados.iloc[idx_test]]
        proporcoes = np.array([len(p) for p in partes]) / len(dados)
        score = float(np.abs(proporcoes - np.array([.70, .15, .15])).sum())
        completas = all(set(p["label"].unique()) == set(CLASSES) for p in partes)
        if completas:
            score -= 10
        if score < melhor_score:
            melhor_score = score
            melhor = (idx_train, idx_val, idx_test, completas, s)
        if completas and np.abs(proporcoes - np.array([.70, .15, .15])).sum() < .08:
            break

    if melhor is None or not melhor[3]:
        raise RuntimeError(
            "Nao foi encontrada divisao por video com as tres classes em "
            "treino, validacao e teste. Verifique a distribuicao por video."
        )
    a, b, c, _, seed_escolhida = melhor
    return dados.iloc[a].copy(), dados.iloc[b].copy(), dados.iloc[c].copy(), seed_escolhida


def carregar_particoes_anteriores(dados):
    nomes = ["treino", "validacao", "teste"]
    caminhos = {
        n: PASTA_PARTICOES_ANTERIORES / f"divisao_{n}.csv" for n in nomes
    }
    if not all(p.exists() for p in caminhos.values()):
        return None

    ids = {}
    for nome, caminho in caminhos.items():
        antigo = pd.read_csv(caminho, encoding="utf-8-sig")
        if "id_video" not in antigo.columns:
            raise ValueError(f"{caminho} nao possui id_video.")
        ids[nome] = set(antigo["id_video"].astype(str).str.strip())

    if (ids["treino"] & ids["validacao"] or
        ids["treino"] & ids["teste"] or
        ids["validacao"] & ids["teste"]):
        raise ValueError("As particoes anteriores possuem videos sobrepostos.")

    ids_atual = set(dados["id_video"].astype(str))
    ids_antigos = set.union(*ids.values())
    if ids_atual != ids_antigos:
        raise ValueError(
            "Os IDs de video do comentarios.csv nao correspondem aos das "
            "particoes anteriores. Confira se esta usando a mesma versao "
            "do dataset e os CSVs de divisao corretos."
        )

    partes = {
        n: dados[dados["id_video"].isin(ids[n])].copy() for n in nomes
    }
    for n, parte in partes.items():
        if set(parte["label"].unique()) != set(CLASSES):
            raise ValueError(f"A particao {n} nao contem as tres classes.")
    return partes["treino"], partes["validacao"], partes["teste"]


particoes = carregar_particoes_anteriores(df)
if particoes is not None:
    train_df, validation_df, test_df = particoes
    metodo_divisao = "IDs de video reutilizados das particoes anteriores"
    print("\nParticoes anteriores encontradas e reutilizadas.")
else:
    train_df, validation_df, test_df, seed_escolhida = dividir_por_video(df)
    metodo_divisao = "nova divisao group-aware por id_video"
    print(
        "\nATENCAO: particoes anteriores nao encontradas. "
        "Foi criada uma nova divisao por video."
    )

for nome, parte in [
    ("treino", train_df), ("validacao", validation_df), ("teste", test_df)
]:
    parte.to_csv(
        PASTA_RESULTADOS / f"divisao_{nome}.csv",
        index=False, encoding="utf-8-sig"
    )

# Verifica ausencia de vazamento entre videos.
g_treino = set(train_df["id_video"])
g_val = set(validation_df["id_video"])
g_teste = set(test_df["id_video"])
assert g_treino.isdisjoint(g_val)
assert g_treino.isdisjoint(g_teste)
assert g_val.isdisjoint(g_teste)

def mostrar_parte(nome, parte):
    print(f"\n{nome}: {len(parte)} comentarios; {parte['id_video'].nunique()} videos")
    print(
        parte["label"].map(ID2LABEL)
        .value_counts().reindex(NOMES_CLASSES, fill_value=0).to_dict()
    )

print("\nDivisao:", metodo_divisao)
mostrar_parte("TREINO", train_df)
mostrar_parte("VALIDACAO", validation_df)
mostrar_parte("TESTE", test_df)

with open(PASTA_RESULTADOS / "configuracao.json", "w", encoding="utf-8") as f:
    json.dump({
        "divisao": metodo_divisao,
        "sementes_treino": SEMENTES_TREINO,
        "estrategias": ESTRATEGIAS,
        "entrada": "somente comentario, sem contexto",
        "max_length": MAX_LENGTH,
        "learning_rate": TAXA_APRENDIZADO,
        "batch_treino": BATCH_TREINO,
        "batch_avaliacao": BATCH_AVALIACAO,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epocas": EPOCAS,
        "filtragem": filtragem,
    }, f, ensure_ascii=False, indent=2)


# ------------------------- TOKENIZACAO E METRICAS -------------------------

def preparar_dataset(parte, tokenizer):
    ds = Dataset.from_pandas(parte[["texto", "label"]], preserve_index=False)
    def tokenizar(exemplos):
        return tokenizer(
            exemplos["texto"], truncation=True, padding=False,
            max_length=MAX_LENGTH
        )
    return ds.map(tokenizar, batched=True)


def calcular_metricas(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }


# ------------------------- TRAINERS ESPECIFICOS -------------------------

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights precisa ser informado.")
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(
            outputs.logits.reshape(-1, model.config.num_labels),
            labels.reshape(-1)
        )
        return (loss, outputs) if return_outputs else loss


class OversamplingTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        # O Dataset conserva a coluna original como "label"; o Trainer a
        # converte para "labels" somente quando prepara os lotes.
        coluna_rotulo = "label" if "label" in ds.column_names else "labels"
        if coluna_rotulo not in ds.column_names:
            raise ValueError(
                "Coluna de rotulos nao encontrada. "
                f"Colunas disponiveis: {ds.column_names}"
            )
        labels = np.asarray(ds[coluna_rotulo], dtype=np.int64)
        classes, counts = np.unique(labels, return_counts=True)
        class_weights = {int(c): 1.0 / float(n) for c, n in zip(classes, counts)}
        sample_weights = [class_weights[int(y)] for y in labels]
        generator = torch.Generator()
        generator.manual_seed(int(self.args.data_seed or self.args.seed))
        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )


# ------------------------- TREINAMENTO E AVALIACAO -------------------------

def executar_experimento(nome_modelo, id_modelo, estrategia, seed_treino):
    pasta = (
        PASTA_RESULTADOS / estrategia / nome_modelo / f"semente_{seed_treino}"
    )
    pasta.mkdir(parents=True, exist_ok=True)

    # Retoma execucoes concluidas: reaproveita metricas.json em vez de
    # repetir treinamentos que ja terminaram antes de uma interrupcao.
    arquivo_metricas_existente = pasta / "metricas.json"
    if arquivo_metricas_existente.exists():
        try:
            with open(arquivo_metricas_existente, "r", encoding="utf-8") as f:
                dados_salvos = json.load(f)
            relatorio_salvo = dados_salvos["metricas_teste"]
            print(f"Execucao ja concluida; reutilizando: {pasta}")
            return {
                "modelo": nome_modelo,
                "estrategia": estrategia,
                "semente_treino": seed_treino,
                "accuracy": float(relatorio_salvo["accuracy"]),
                "precision_macro": float(relatorio_salvo["macro avg"]["precision"]),
                "recall_macro": float(relatorio_salvo["macro avg"]["recall"]),
                "f1_macro": float(relatorio_salvo["macro avg"]["f1-score"]),
                "f1_negativo": float(relatorio_salvo["NEGATIVO"]["f1-score"]),
                "f1_neutro": float(relatorio_salvo["NEUTRO"]["f1-score"]),
                "f1_positivo": float(relatorio_salvo["POSITIVO"]["f1-score"]),
            }
        except (KeyError, ValueError, json.JSONDecodeError) as erro:
            print(
                "Metricas existentes incompletas; o treinamento sera refeito. "
                f"Detalhe: {erro}"
            )

    print("\n" + "=" * 72)
    print("MODELO:", nome_modelo, "| ESTRATEGIA:", estrategia, "| SEED:", seed_treino)
    print("=" * 72)

    random.seed(seed_treino)
    np.random.seed(seed_treino)
    torch.manual_seed(seed_treino)
    set_seed(seed_treino)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_treino)

    tokenizer = AutoTokenizer.from_pretrained(id_modelo)
    config = AutoConfig.from_pretrained(id_modelo)
    limite = getattr(config, "max_position_embeddings", MAX_LENGTH)
    if MAX_LENGTH > limite:
        raise ValueError(f"MAX_LENGTH {MAX_LENGTH} excede limite {limite}.")
    del config

    ds_train = preparar_dataset(train_df, tokenizer)
    ds_val = preparar_dataset(validation_df, tokenizer)
    ds_test = preparar_dataset(test_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        id_modelo, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    contagem = train_df["label"].value_counts().reindex(CLASSES, fill_value=0)
    if (contagem == 0).any():
        raise ValueError(f"Classe ausente no treino: {contagem.to_dict()}")

    pesos = compute_class_weight(
        class_weight="balanced", classes=CLASSES,
        y=train_df["label"].to_numpy()
    )
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

    print("Contagem no treino:", {
        ID2LABEL[int(k)]: int(v) for k, v in contagem.items()
    })
    print("Pesos calculados:", {
        ID2LABEL[i]: round(float(pesos[i]), 4) for i in range(3)
    })

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    args = TrainingArguments(
        output_dir=str(pasta / "checkpoints"),
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

    comuns = dict(
        model=model, args=args, train_dataset=ds_train,
        eval_dataset=ds_val, data_collator=collator,
        processing_class=tokenizer, compute_metrics=calcular_metricas
    )

    if estrategia == "original":
        trainer = Trainer(**comuns)
    elif estrategia == "weighted_cross_entropy":
        trainer = WeightedLossTrainer(
            **comuns, class_weights=pesos_tensor
        )
    elif estrategia == "oversampling":
        trainer = OversamplingTrainer(**comuns)
    else:
        raise ValueError(f"Estrategia desconhecida: {estrategia}")

    trainer.train()
    trainer.save_model(str(pasta / "modelo_final"))
    tokenizer.save_pretrained(str(pasta / "modelo_final"))

    # O teste permanece com a distribuicao original: nao aplicar sampler aqui.
    resultado_teste = trainer.evaluate(ds_test)
    pred = trainer.predict(ds_test)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)

    relatorio = classification_report(
        y_true, y_pred, labels=[0, 1, 2],
        target_names=NOMES_CLASSES, output_dict=True, zero_division=0
    )
    relatorio_txt = classification_report(
        y_true, y_pred, labels=[0, 1, 2],
        target_names=NOMES_CLASSES, digits=4, zero_division=0
    )
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    (pasta / "relatorio_classificacao.txt").write_text(
        relatorio_txt, encoding="utf-8"
    )
    pd.DataFrame(
        matriz,
        index=[f"real_{n}" for n in NOMES_CLASSES],
        columns=[f"previsto_{n}" for n in NOMES_CLASSES]
    ).to_csv(pasta / "matriz_confusao.csv", encoding="utf-8-sig")

    previsoes = test_df[
        ["id_video", "comentario_original", "comentario", "label"]
    ].copy()
    previsoes["rotulo_real"] = previsoes["label"].map(ID2LABEL)
    previsoes["label_previsto"] = y_pred
    previsoes["rotulo_previsto"] = previsoes["label_previsto"].map(ID2LABEL)
    previsoes["correto"] = previsoes["label"] == previsoes["label_previsto"]
    previsoes.to_csv(
        pasta / "predicoes_teste.csv", index=False, encoding="utf-8-sig"
    )

    metadados = {
        "modelo": nome_modelo,
        "modelo_pre_treinado": id_modelo,
        "estrategia": estrategia,
        "semente_treino": seed_treino,
        "entrada": "somente comentario, sem contexto",
        "contagem_classes_treino": {
            ID2LABEL[int(k)]: int(v) for k, v in contagem.items()
        },
        "pesos_calculados": {
            ID2LABEL[i]: float(pesos[i]) for i in range(3)
        },
        "metricas_teste": relatorio,
        "metricas_teste_trainer": {
            k: float(v) for k, v in resultado_teste.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        },
        "matriz_confusao": matriz.tolist(),
    }
    with open(pasta / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print("\nRelatorio de classificacao:\n", relatorio_txt)
    print("Matriz de confusao:\n", matriz)

    resumo = {
        "modelo": nome_modelo,
        "estrategia": estrategia,
        "semente_treino": seed_treino,
        "accuracy": float(relatorio["accuracy"]),
        "precision_macro": float(relatorio["macro avg"]["precision"]),
        "recall_macro": float(relatorio["macro avg"]["recall"]),
        "f1_macro": float(relatorio["macro avg"]["f1-score"]),
        "f1_negativo": float(relatorio["NEGATIVO"]["f1-score"]),
        "f1_neutro": float(relatorio["NEUTRO"]["f1-score"]),
        "f1_positivo": float(relatorio["POSITIVO"]["f1-score"]),
    }

    del trainer, model, tokenizer, ds_train, ds_val, ds_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resumo


# ------------------------- EXECUTAR TODAS AS COMBINACOES -------------------------

modelos_execucao = MODELOS if EXECUTAR_DOIS_MODELOS else {
    "bertimbau": MODELOS["bertimbau"]
}
total = len(modelos_execucao) * len(ESTRATEGIAS) * len(SEMENTES_TREINO)
print(f"\nTotal de treinamentos: {total}")

resumos = []
for estrategia in ESTRATEGIAS:
    for seed in SEMENTES_TREINO:
        for nome, identificador in modelos_execucao.items():
            resumos.append(
                executar_experimento(nome, identificador, estrategia, seed)
            )

comparacao = pd.DataFrame(resumos).sort_values(
    ["modelo", "estrategia", "semente_treino"]
).reset_index(drop=True)

comparacao.to_csv(
    PASTA_RESULTADOS / "resultados_todas_execucoes.csv",
    index=False, encoding="utf-8-sig"
)

metricas = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "f1_negativo", "f1_neutro", "f1_positivo"
]
agrupado = comparacao.groupby(["modelo", "estrategia"])[metricas].agg(
    ["mean", "std"]
)
agrupado.columns = [
    f"{metrica}_{estatistica}"
    for metrica, estatistica in agrupado.columns
]
agrupado = agrupado.reset_index()

agrupado.to_csv(
    PASTA_RESULTADOS / "media_desvio_padrao.csv",
    index=False, encoding="utf-8-sig"
)

print("\nRESULTADOS INDIVIDUAIS")
print(comparacao.to_string(index=False))
print("\nMEDIA E DESVIO-PADRAO")
print(agrupado.to_string(index=False))

arquivo_zip = "/content/resultados_balanceamento.zip"
if os.path.exists(arquivo_zip):
    os.remove(arquivo_zip)

with zipfile.ZipFile(arquivo_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for arquivo in PASTA_RESULTADOS.rglob("*"):
        if arquivo.is_file():
            z.write(
                arquivo,
                arcname=str(arquivo.relative_to(PASTA_RESULTADOS.parent))
            )

print("\nConcluido.")
print("Resultados:", PASTA_RESULTADOS)
print("ZIP:", arquivo_zip)
print("Para baixar:")
print("from google.colab import files")
print("files.download('/content/resultados_balanceamento.zip')")
