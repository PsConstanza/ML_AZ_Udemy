# Regresión Lineal Múltiple
# Importar el dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[,2:3]

# Codificar las variables categóricas
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1,2,3))

# Dividir data set en conjunto de entrenamiento y testing
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de Regresión Lineal Múltiple al conjunto de Entrenamiento
regression = lm(formula = Profit ~ .,
                data = training_set)

# Predecir los resultados con el conjunto de Testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo óptimo con la eliminación hacia atrás
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State ,
                data = dataset)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
                data = dataset)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend ,
                data = dataset)

regression = lm(formula = Profit ~ R.D.Spend ,
                data = dataset)

summary(regression)

# install.packages("https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.2.tar.gz",repos=NULL, type="source")