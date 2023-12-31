# Clustering Jer�rquico

# Importar los datos
dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

# Utilizar el dendrograma para encontrar el nro. �ptimo de clusters
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = 'Dendrograma',
     xlab = 'Clientes del Centro Comercial',
     ylab = 'Distancia Euclidea')

# Ajustar el Clstering Jer�rquico al dataset
hc = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
y_hc = cutree(hc, k = 5)

# Visualizar los clusters
# install.packages('cluster')
library(cluster)
clusplot(X, 
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE, 
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = 'Clustering Clientes',
         xlab = 'Ingresos Anuales (en miles de USD$)',
         ylab = 'Puntuaci�n (1-100)')


