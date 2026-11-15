import os
import re
import sys
import json
import csv
from datetime import datetime

import requests

# ==============================
# CONFIGURAÇÃO
# ==============================

API_KEY = os.environ.get("YOUTUBE_API_KEY", "COLOQUE_SUA_API_KEY_AQUI")

COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

MAX_RESULTS = 100  # máximo permitido por página pela API


# ==============================
# EXTRAÇÃO DO VIDEO_ID A PARTIR DO LINK
# ==============================

def extrair_video_id(url_ou_id):
    """
    Aceita diversos formatos de link do YouTube (ou o próprio ID) e retorna
    o video_id.

    Formatos suportados:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - VIDEO_ID puro (11 caracteres)
    """
    url_ou_id = url_ou_id.strip()

    padroes = [
        r"(?:v=|/videos/|embed/|shorts/|youtu\.be/)([A-Za-z0-9_-]{11})",
    ]

    for padrao in padroes:
        match = re.search(padrao, url_ou_id)
        if match:
            return match.group(1)

    # Se não bateu com nenhum padrão de URL, assume que já é o ID puro
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_ou_id):
        return url_ou_id

    raise ValueError(f"Não foi possível extrair o video_id de: {url_ou_id}")


# ==============================
# METADADOS DO VÍDEO (**opcional)
# ==============================

def obter_metadados_video(video_id):
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": API_KEY,
    }
    response = requests.get(VIDEOS_URL, params=params)
    response.raise_for_status()
    data = response.json()

    items = data.get("items", [])
    if not items:
        return {}

    snippet = items[0].get("snippet", {})
    stats = items[0].get("statistics", {})

    return {
        "titulo": snippet.get("title"),
        "canal": snippet.get("channelTitle"),
        "data_publicacao": snippet.get("publishedAt"),
        "views": stats.get("viewCount"),
        "likes_video": stats.get("likeCount"),
        "total_comentarios_reportado": stats.get("commentCount"),
    }


# ==============================
# COLETA DE COMENTÁRIOS
# ==============================

def coletar_comentarios(video_id):
    comentarios = []
    next_page_token = None

    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": MAX_RESULTS,
            "key": API_KEY,
            "textFormat": "plainText",
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(COMMENTS_URL, params=params)

        if response.status_code != 200:
            erro = response.json().get("error", {})
            mensagem = erro.get("message", response.text)
            print(f"Erro da API ({response.status_code}): {mensagem}")

            # Comentários desabilitados no vídeo (403) ou vídeo não encontrado
            if response.status_code in (403, 404):
                break
            raise requests.HTTPError(mensagem)

        data = response.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comentario = {
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "autor": snippet.get("authorDisplayName"),
                "comentario": snippet.get("textDisplay"),
                "likes": snippet.get("likeCount"),
                "data_publicacao": snippet.get("publishedAt"),
                "total_respostas": item["snippet"].get("totalReplyCount", 0),
            }

            comentarios.append(comentario)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comentarios


# ==============================
# EXPORTAÇÃO PARA O FORMATO DO LABEL STUDIO
# ==============================

def exportar_label_studio(comentarios, caminho_saida):
    """
    O Label Studio, para importação de tarefas, espera uma lista de objetos
    JSON no formato:

        [
            {"data": {"text": "...", "campo_extra": "..."}},
            {"data": {"text": "...", "campo_extra": "..."}}
        ]

    O campo usado no template de rotulação (ex: <Text name="text" value="$text"/>)
    deve corresponder à chave dentro de "data" (aqui, "text").
    """
    tarefas = []

    for c in comentarios:
        tarefas.append({
            "data": {
                "text": c["comentario"],
                "comment_id": c["comment_id"],
                "video_id": c["video_id"],
                "autor": c["autor"],
                "likes": c["likes"],
                "data_publicacao": c["data_publicacao"],
            }
        })

    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(tarefas, f, ensure_ascii=False, indent=2)


# ==============================
# SALVAR CSV / JSON BRUTOS (para arquivamento/backup)
# ==============================

def salvar_dados_brutos(comentarios, video_id, timestamp):
    json_file = f"comentarios_{video_id}_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(comentarios, f, ensure_ascii=False, indent=2)

    csv_file = f"comentarios_{video_id}_{timestamp}.csv"
    if comentarios:
        keys = comentarios[0].keys()
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comentarios)

    return json_file, csv_file


# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================

def main():
    if API_KEY == "COLOQUE_SUA_API_KEY_AQUI":
        print("Defina sua API key na variável de ambiente YOUTUBE_API_KEY "
              "ou substitua diretamente na variável API_KEY do script.")
        sys.exit(1)

    if len(sys.argv) > 1:
        link_video = sys.argv[1]
    else:
        link_video = input("Cole o link (ou ID) do vídeo do YouTube: ")

    video_id = extrair_video_id(link_video)
    print(f"Video ID identificado: {video_id}")

    print("Buscando metadados do vídeo...")
    metadados = obter_metadados_video(video_id)
    if metadados:
        print(f"Título: {metadados.get('titulo')}")
        print(f"Canal: {metadados.get('canal')}")
        print(f"Comentários reportados pela API: {metadados.get('total_comentarios_reportado')}")

    print("Coletando comentários...")
    comentarios = coletar_comentarios(video_id)
    print(f"Total de comentários coletados: {len(comentarios)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file, csv_file = salvar_dados_brutos(comentarios, video_id, timestamp)

    label_studio_file = f"label_studio_{video_id}_{timestamp}.json"
    exportar_label_studio(comentarios, label_studio_file)

    print("\nColeta finalizada!")
    print(f"Backup bruto (JSON): {json_file}")
    print(f"Backup bruto (CSV): {csv_file}")
    print(f"Arquivo para importar no Label Studio: {label_studio_file}")


if __name__ == "__main__":
    main()
