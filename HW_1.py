import numpy as np

# Se cargan los datos
y_total = np.genfromtxt("msd_genre_dataset.txt",dtype =None,usecols=0,comments=None,delimiter=',',skip_header=10)
x_total = np.genfromtxt("msd_genre_dataset.txt",comments=None,delimiter=',',skip_header=10)[:,4:]

generos = [b'metal',b'punk',b'dance and electronica']

y_ent = []
x_raw = []
for i in range(0,y_total.size):
    if y_total[i] in generos:
        y_ent.append(y_total[i])
        x_raw.append(x_total[i])

y_ent = np.asarray(y_ent)
x_raw = np.asarray(x_raw)

# Preprocesamiento
x_prep = (x_raw - np.mean(x_raw,axis=0))/np.std(x_raw,axis=0)
x_matriz_cov = np.cov(x_prep)

x_matriz_cov.shape

u,s,v = np.linalg.svd(x_matriz_cov)
