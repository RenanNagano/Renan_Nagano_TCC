Os experimentos de Fine-Tuning foram executados no colab.research.google.com
Para realizá-los, basta subir todos os arquivos da pasta para o ambiente e executar o .py

#No Colab, selecione T4 GPU em "runtime type"

#Comandos

#1. Upload de todos os arquivos da pasta do experimento escolhido
from google.colab import files
arquivos = files.upload()

#2. Instale as bibliotecas 
!pip install -q transformers datasets accelerate scikit-learn emoji pandas numpy

#3. Execute o .py do experimento escolhido
!python /content/ARQUIVO_DO_EXPERIMENTO.py

