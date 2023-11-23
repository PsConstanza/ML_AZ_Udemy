# Natural Language Processing

# Importar Dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

# Limpieza de Texto
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
# Consultar el primer Elemento del Corpus
# as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = 'en')) 
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Crear el Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar a variable de clasificación como factor
dataset$Liked = factor(dataset$Liked,
                       levels = c(0,1))


# 01 Regresión Logística
# Dividir los datos en conjunto de entrenamiento y conjunto de testing
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE )
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresión logística con el conjunto de datos
classifier = glm(formula = Liked ~ .,
                 data = training_set, 
                 family = binomial)

# Predicción de los resultados con el conjunto de testing 
prob_pred = predict(classifier, type = 'response',
                    newdata = testing_set[,-692])
y_pred = ifelse(prob_pred>0.5, 1, 0)

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 02 KNN
# Ajustar el clasificador con el conjunto de datos
# install.packages('class')
library(class)
y_pred = knn(train = training_set[,-692],
             test = testing_set[,-692],
             cl = training_set[,692],
             k = 5)

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 03 SVM
# Ajustar el SVM con el conjunto de datos
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 04 Kernel SVM
# Ajustar el kernel SVM con el conjunto de datos
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 05 Naive Bayes
# Ajustar el Naive Bayes con el conjunto de datos
# install.packages('e1071')
library(e1071)
classifier = naiveBayes( x = training_set[,-692],
                         y = training_set$Liked)

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 06 Decission Tree
# Ajustar el clasificador de árbol de decisión con el conjunto de datos
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set)

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692],
                 type = 'class')

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 07 Random Forest
# Ajustar el clasificador Random Forest con el conjunto de datos
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 08 Cart
# Ajustar el clasificador de CART con el conjunto de datos
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set,
                   method = 'class',
                   control = rpart.control(cp = 0.01))

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692],
                 type = 'class')

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 09 C5.0
# Ajustar el clasificador de C5.0 con el conjunto de datos
# install.packages('C50')
library(C50)
classifier = C5.0(formula = Liked ~ .,
                  data = training_set,
                  trials = 10)

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692],
                 type = 'class')

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


# 10 Max. Entropy
# Ajustar el clasificador de max ent con el conjunto de datos
# install.packages("nnet")
library(nnet)
classifier = multinom(formula = Liked ~ .,
                  data = training_set,
                  MaxNWts = 1000)

# Predicción de los resultados con el conjunto de testing 
y_pred = predict(classifier, newdata = testing_set[,-692],
                 type = 'class')

# Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)