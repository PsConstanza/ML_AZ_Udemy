# �rbol de Decisi�n  para Regresi�n

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# Dividir data set en conjunto de entrenamiento y testing
# install.packages("caTools")
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de regresi�n on el conjunto de datos
# install.packages("rpart")
library(rpart)
regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1) )


# Predicci�n de nuevos resultados con �rbol de Regresi�n
y_pred = predict(regression, newdata = data.frame(Level = 6.5))
#y_pred = predict(regression, newdata = data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3,
#Level4 = 6.5^4, Level11 = 6.5^11))

# Visualizaci�n del modelo de �rbol de regresi�n
# install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary ),
             color = "red") + 
  geom_line(aes(x = dataset$Level , y = predict(regression, newdata = data.frame(Level = dataset$Level) )),
            color = "blue") + 
  ggtitle("Predicci�n con �rbol de Decisi�n(Modelo de Regresi�n)") + 
  xlab("Nivel del Empleado") + 
  ylab("Sueldo (en USD$)")

# Visualizaci�n del modelo �rbol de regresi�n con grid
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary ),
             color = "red") + 
  geom_line(aes(x = x_grid , y = predict(regression, newdata = data.frame(Level = x_grid) )),
            color = "blue") + 
  ggtitle("Predicci�n con �rbol de Decisi�n(Modelo de Reg)") + 
  xlab("Nivel del Empleado") + 
  ylab("Sueldo (en USD$)")

