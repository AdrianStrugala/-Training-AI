from scipy.stats import pearsonr

#cov = 1 -> wraz ze wzrostem jednej zmiennej rosnie druga
#cov = -1 -> jezeli jeden wekttor maleje, to maleje tez drugi
#korelacja pearsona mowi tylko o trendzie (a nie o szybkosci wzrostu)

#lepsza graficznie jest heatmap z sklearn :) 

v1 = [1,2,3,4]
v2 = [2,4,6,9]

print(pearsonr(v1, v2))


#algorytm PCA sluzy do zmiejszania wymiarow danych (liczby kolumn)