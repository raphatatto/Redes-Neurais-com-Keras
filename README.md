#  Checkpoint 2 – Redes Neurais com Keras
  **Integrantes**
- Raphaela Oliveira Tatto - 554983
- Otavio Miklos - 554513
- Luciayla Kawakami - 557987
- Tiago Ribeiro Capela - 558021
##  Estrutura do Repositório

```
CP2-IOT/
│
├── ex1_classificacao.py      # Exercício 1 - Classificação (Wine Dataset)
├── ex2_regressao.py          # Exercício 2 - Regressão (California Housing)
├── requirements.txt          # Lista de dependências para instalação
└── README.md                 # Este arquivo
```

---

##  Como Executar os Experimentos

1. **Clonar o repositório**
```bash
git clone https://github.com/raphatatto/Redes-Neurais-com-Keras.git
cd cp2-iot
```

2. **Instalar as dependências**
```bash
pip install -r requirements.txt
```

3. **Rodar os scripts**
```bash
python ex1_classificacao.py
python ex2_regressao.py
```

---

##  Exercício 1 – Classificação Multiclasse (Wine Dataset)

- **Configuração da Rede Neural:**
  - 2 camadas ocultas com 32 neurônios cada, ativação **ReLU**
  - Camada de saída com 3 neurônios, ativação **Softmax**
  - Função de perda: `categorical_crossentropy`
  - Otimizador: **Adam**
  - Treinada por 50 épocas, batch size = 8

- **Resultados:**
  - **Acurácia Rede Neural:** 1.0000
  - **Acurácia RandomForest:** 1.0000

> **Conclusão:** Ambas as abordagens atingiram 100% de acurácia no conjunto de teste.  
> Isso indica que o dataset é bem separável e que tanto a rede neural quanto o RandomForest  
> conseguem aprender a fronteira de decisão de forma eficaz.

---

##  Exercício 2 – Regressão (California Housing Dataset)

- **Configuração da Rede Neural:**
  - 3 camadas ocultas (64, 32, 16 neurônios) com ativação **ReLU**
  - Camada de saída com 1 neurônio, ativação **Linear**
  - Função de perda: `mse`
  - Otimizador: **Adam**
  - Treinada por 50 épocas, batch size = 32

- **Resultados:**
  - **RMSE Rede Neural:** 0.8619
  - **RMSE LinearRegression:** 0.7456

> **Conclusão:** A regressão linear apresentou melhor desempenho que a rede neural,  
> provavelmente devido à natureza quase linear das relações entre as variáveis do dataset.  
> Uma rede neural poderia superar este resultado com mais ajustes de hiperparâmetros  
> (maior número de épocas, dropout, normalização de entrada, etc.).

---

## 📊 Comparação Geral

| Exercício | Modelo Clássico      | Métrica  | Resultado | Rede Neural | Resultado |
|----------|---------------------|---------|-----------|------------|-----------|
| 1 – Classificação | RandomForestClassifier | Acurácia | **1.0000** | Keras NN | **1.0000** |
| 2 – Regressão     | LinearRegression      | RMSE     | **0.7456** | Keras NN | 0.8619 |

---

##  Conclusão Final
- **Classificação:** Tanto modelos clássicos quanto redes neurais têm desempenho perfeito neste dataset.
- **Regressão:** Modelos lineares ainda superam redes neurais neste conjunto, mas o uso de redes permite explorar relações não-lineares se houver maior complexidade ou mais dados.

---

##  Dependências

Veja o arquivo [`requirements.txt`](./requirements.txt):

```
tensorflow>=2.15.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
matplotlib>=3.8.0
```

---

