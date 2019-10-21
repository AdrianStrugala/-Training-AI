import numpy as np

# numpy - statystyka, obliczenia numeryczne
# pandas - in memory database
# sklearn - tradycyjne algorytmy ML

#sieci neuronowe:
# pytorch
# keras
# tensorflow 

v1 = np.array([1, 3, 5])
v2 = np.array([4, -3, -1])

#Iloczyn skalarny
print(np.dot(v1, v2))

#Dlugosc wektora
print(np.linalg.norm([1,1]))

#Macierze
m1 = np.matrix([[1,1], [2,3]])
m2 = np.matrix([[1,1], [2,3]])

scalar = 5

print(m2 * m1)

print(scalar * m1)

#Transpozycja
print(m1.T)

#Wymiar macierzy
print(m1.shape)


#Range
dat = np.array([range(1,10)])

print(dat)

#Srednia
print(np.mean(dat))

#Mediana
print(np.median(dat))

#Wariancja (srednia odleglosc od sredniej)
print(np.var(dat))

#Odchylenei standardowe (wariancja do kwadratu)
print(np.std(dat))
