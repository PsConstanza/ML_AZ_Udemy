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

# Random Forest
# Codificar a variable de clasificación como factor
dataset$Liked = factor(dataset$Liked,
                           levels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y conjunto de testing
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE )
testing_set = subset(dataset, split == FALSE)
 
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
