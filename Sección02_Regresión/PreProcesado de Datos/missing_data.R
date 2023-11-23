#Plantilla para Pre Procesado de datos - Datos Faltantes

#Importar el dataset
dataset = read.csv('Data.csv')

#Tratamiendo de los valores NAs
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)