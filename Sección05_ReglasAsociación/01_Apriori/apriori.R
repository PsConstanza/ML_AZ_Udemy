# Apriori

# Preprocesado de Datos
# install.packages('arules')
library(arules)

dataset = read.csv('Market_Basket_Optimisation.csv',header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv',
                            sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

# Entrenar el algoritmo Apriori con el dataset
rules = apriori(data = dataset, parameter = list(support = 0.004 , confidence = 0.2 ))

#Visualización de los resultados
inspect(sort(rules, by = 'lift')[1:10])


# ------------------------------------------------------------------------
# GOAL: show how to create html widgets with transaction rules
# ------------------------------------------------------------------------

# libraries --------------------------------------------------------------
library(arules)
install.packages('arulesViz')
library(arulesViz)

# data -------------------------------------------------------------------
path <- "/Users/Ckonny/Desktop/ML/Sección05_ReglasAsociación/01_Apriori/"
trans <- read.transactions(
  file = paste0(path, "Market_Basket_Optimisation.csv"),
  sep = ",",
  rm.duplicates = TRUE
)

# apriori algoirthm ------------------------------------------------------
rules <- apriori(
  data = trans,
  parameter = list(support = 0.004, confidence = 0.2)
)

# visualizations ---------------------------------------------------------
plot(rules, method = "graph", engine = "htmlwidget")
