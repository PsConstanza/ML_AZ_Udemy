#Plantilla para Pre Procesado de datos - Datos Categóricos

#Importar el dataset
dataset = read.csv('Data.csv')

# Codificar las variables categóricas
# Country
dataset$Country = factor(dataset$Country,
                         levels = c("France","Spain","Germany"),
                         labels = c(1,2,3))
# Purchase
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No","Yes"),
                           labels = c(0,1))