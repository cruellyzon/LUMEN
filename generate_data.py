# generate_data.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Criar a pasta data se não existir
os.makedirs("data", exist_ok=True)

# Semente para reprodução
np.random.seed(42)

# Configurações dos dados
n_samples = 500000
n_genetic = 10
n_environmental = 8
n_relativistic = 6

# Gerando dados aleatórios normalizados
X_gen = np.random.normal(0, 1, size=(n_samples, n_genetic))
X_amb = np.random.normal(0, 1, size=(n_samples, n_environmental))
X_rel = np.random.normal(0, 1, size=(n_samples, n_relativistic))

# Pesos ocultos para simular influência de cada domínio
weights_gen = np.random.uniform(0.5, 1.5, size=n_genetic)
weights_amb = np.random.uniform(0.3, 1.0, size=n_environmental)
weights_rel = np.random.uniform(0.7, 1.2, size=n_relativistic)

# Combinação linear + ruído
signal = (
    X_gen @ weights_gen +
    X_amb @ weights_amb +
    X_rel @ weights_rel +
    np.random.normal(0, 2, size=n_samples)
)

# Transformação para valores entre 0 e 1
probabilities = 1 / (1 + np.exp(-0.01 * signal))

# Geração dos rótulos binários
y = (probabilities > 0.5).astype(int)

# Divisão em treino/teste (80/20)
X_gen_train, _, X_amb_train, _, X_rel_train, _, y_train, _ = train_test_split(
    X_gen, X_amb, X_rel, y, test_size=0.2, random_state=42
)

# Empilhamento horizontal
X_total = np.hstack([X_gen_train, X_amb_train, X_rel_train])
columns = (
    [f"GEN_{i+1}" for i in range(n_genetic)] +
    [f"AMB_{i+1}" for i in range(n_environmental)] +
    [f"REL_{i+1}" for i in range(n_relativistic)]
)

df = pd.DataFrame(X_total, columns=columns)
df["TARGET"] = y_train

# Salvando em CSV
df.to_csv("data/data.csv", index=False)

print("✅ Dados gerados com sucesso: data/data.csv")
print("🔍 Shape:", df.shape)
