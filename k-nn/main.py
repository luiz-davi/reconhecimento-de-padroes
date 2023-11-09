import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Carregue o conjunto de dados Iris
iris = datasets.load_iris()
dados = iris.data
classificacao = iris.target

testeIndices = []
treinoIndices = []
treinoClasses = []  # Lista para armazenar as classes dos elementos de treino

# Para cada classe (0, 1 e 2), escolha aleatoriamente um exemplo para teste
# e dois exemplos para treino
for classe in range(3):
    indiceClasse = np.where(classificacao == classe)[0]
    testeIndices.append(np.random.choice(indiceClasse, size=1, replace=False))
    treinoEscolhido = np.random.choice(indiceClasse, size=2, replace=False)
    treinoIndices.extend(treinoEscolhido)
    treinoClasses.extend([classe, classe])  # Repetir a classe duas vezes para os elementos de treino

# Converta as listas de índices e classes em arrays NumPy
testeIndices = np.array(testeIndices)
treinoIndices = np.array(treinoIndices)
treinoClasses = np.array(treinoClasses)

print(testeIndices, treinoIndices)

# Extraia os exemplos de teste e treino
dadosTeste = dados[testeIndices]
dadosTreino = dados[treinoIndices]
classesTreino = treinoClasses  # Classes correspondentes aos elementos de treino

# Inicialize uma matriz de distâncias
matrixDistancia = np.zeros((3, 6))

# Calcule as distâncias euclidianas
for i in range(3):
    for j in range(6):
        matrixDistancia[i, j] = np.sqrt(np.sum((dadosTeste[i] - dadosTreino[j]) ** 2))

print("Matriz de Distâncias:")
print(matrixDistancia)
print("Classes dos elementos de treino:")
print(classesTreino)