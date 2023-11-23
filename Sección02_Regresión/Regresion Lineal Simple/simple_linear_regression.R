# Regresión Lineal Simple

# Importar el dataset
dataset = read.csv('Salary_Data.csv')

# Dividir data set en conjunto de entrenamiento y testing
# install.packages("caTools")

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el MRLS con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predecir resultados con el conjunto de test
y_pred = predict(regressor, newdata = testing_set)

# Visualización de resultados en el Conj. de entrenamiento
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, 
                 y = training_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle("Sueldo vs Años de Experiencia (Conj. de Entrenamiento)") + 
  xlab("Años de Experiencia") + 
  ylab("Sueldo (en USD$)")

# Visualización de resultados en el Conj. de testing
library(ggplot2)
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience, 
                 y = testing_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle("Sueldo vs Años de Experiencia (Conj. de Testing)") + 
  xlab("Años de Experiencia") + 
  ylab("Sueldo (en USD$)")