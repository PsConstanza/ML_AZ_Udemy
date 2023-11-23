# Regresión Polinómica

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

# Ajustar el modelo de regresión lineal con el conjunto de datos
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Ajustar el modelo de regresión polinómica con el conjunto de datos
dataset$Level2= dataset$Level^2
dataset$Level3= dataset$Level^3
dataset$Level4= dataset$Level^4
dataset$Level11= dataset$Level^11
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualización del modelo lineal
# install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary ),
             color = "red") + 
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset )),
            color = "blue") + 
  ggtitle("Predicción Polinómica del Sueldo en Función del Nivel del Empleado") + 
  xlab("Nivel del Empleado") + 
  ylab("Sueldo (en USD$)")

# Visualización del modelo polinómico
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary ),
             color = "red") + 
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid, Level2 = x_grid^2,
                                                                       Level3 = x_grid^3, Level4 = x_grid^4,
                                                                       Level11 = x_grid^11) )),
            color = "blue") + 
  ggtitle("Predicción Lineal del Sueldo en Función del Nivel del Empleado") + 
  xlab("Nivel del Empleado") + 
  ylab("Sueldo (en USD$)")

# Predicción de nuevos resultados con Regresión Lineal
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Predicción de nuevos resultados con Regresión Polinómica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4,
                                                     Level11 = 6.5^11))