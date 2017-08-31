rm(list=ls())
setwd("C:/Users/NikhilS/Desktop/assign/medicine")
install.packages("syuzhet")
install.packages("tm")
install.packages("stringr")
install.packages("caret")
install.packages("xgboost")
install.packages("Matrix")
library(data.table)
library(syuzhet) 
library(tm)
library(stringr)
library(caret)
library(xgboost)
library(Matrix)






# Load CSV files
train_text <- do.call(rbind,strsplit(readLines('training_text'),'||',fixed=T))
train_text <- as.data.table(train_text)
colnames(train_text) <- c("ID", "Text")
train_text$ID <- as.numeric(train_text$ID)

test_text <- do.call(rbind,strsplit(readLines('test_text'),'||',fixed=T))
test_text <- as.data.table(test_text)
colnames(test_text) <- c("ID", "Text")
test_text$ID <- as.numeric(test_text$ID)

train <- fread("training_variants", sep=",", stringsAsFactors = T)
test <- fread("test_variants", sep=",", stringsAsFactors = T)
train_text <- train_text[-1,]
test_text <- test_text[-1,]
#missing values
sum(is.na(train))
sum(is.na(test))

train <- merge(train,train_text,by="ID")
test <- merge(test,test_text,by="ID")
rm(test_text,train_text);gc()

test$Class <- -1
data <- rbind(train,test)
rm(train,test);gc()

# text features
data$nchar <- as.numeric(nchar(data$Text))
data$nwords <- as.numeric(str_count(data$Text, "\\S+"))

# TF-IDF
txt <- Corpus(VectorSource(data$Text))
txt <- tm_map(txt, stripWhitespace)
txt <- tm_map(txt, content_transformer(tolower))
txt <- tm_map(txt, removePunctuation)
txt <- tm_map(txt, removeWords, stopwords("english"))
#txt <- tm_map(txt, stemDocument, language="english")
txt <- tm_map(txt, removeNumbers)
dtm <- DocumentTermMatrix(txt, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.95)
data <- cbind(data, as.matrix(dtm))
rm(dtm,txt);gc()

# labelCount encoding function
labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}


# labelcount encoding for gene and variation to number
data$Gene <- labelCountEncoding(data$Gene)
data$Variation <- labelCountEncoding(data$Variation)

# Sentiment analysis
sentiment <- get_nrc_sentiment(data$Text) 
data <- cbind(data,sentiment) 

# Set seed
set.seed(2016)
cvFoldsList <- createFolds(data$Class[data$Class > -1], k=5, list=TRUE, returnTrain=FALSE)

# sparse matrix
varnames <- setdiff(colnames(data), c("ID", "Class", "Text"))
train_sparse <- Matrix(as.matrix(sapply(data[Class > -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[Class == -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[Class > -1,Class]-1
test_ids <- data[Class == -1,ID]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)

# Params for xgboost
param <- list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              eta = .2,
              gamma = 1,
              max_depth = 5,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

# Cross validation determine optimal amount of rounds
xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1000,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cvFoldsList,
                 early_stopping_rounds = 100
)

# model 
xgb_model <- xgb.train(data = dtrain,
                       params = param,
                       watchlist = list(train = dtrain),
                       nrounds = 43,
                       verbose = 1
                    
)

# importance
names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names,model=xgb_model)
xgb.plot.importance(importance_matrix[1:30,],1)

# Predict 
preds <- as.data.table(t(matrix(predict(xgb_model, dtest), nrow=9, ncol=nrow(dtest))))
colnames(preds) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.table(data.table(ID=test_ids, preds), "submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

