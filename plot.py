import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv(
    "/media/weverton/D/Remote Sensing/Water Quality/mobile_bay/northern-gulf/collect_points/days/21_collect_points_2007-05-06.csv"
)

# Verificar se a coluna 'ChlA' existe no CSV
if "ChlA" in df.columns:
    # Plotar 'ChlA' no eixo X e o índice das linhas no eixo Y
    plt.plot(df["ChlA"])
    plt.xlabel("Índice")
    plt.ylabel("ChlA")
    plt.title("Gráfico do campo ChlA")
    plt.show()
else:
    print("A coluna 'ChlA' não foi encontrada no arquivo CSV.")
