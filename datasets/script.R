#abrir biblioteca
library("tidyverse");
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)


FA <- read.csv("RProjects/FA_1994-2021.csv", sep=";")
FA$OBITO <- factor(FA$OBITO, levels = c('SIM', 'NAO'))
FA <- filter(FA, OBITO %in% c('SIM','NAO'))
#FA$DT_IS <- NULL
#FA$SEXO <- NULL
#FA$MUN_LPI <- NULL
#FA$TOBACCO <- NULL
#FA$CARDIOVASCULAR <- NULL
#FA$ASTHMA <- NULL
#FA$COPD <- NULL
#FA$OBESITY <- NULL
#FA$USMER <- NULL
#FA$PREGNANT <- NULL
#FA$SEX <- NULL
#FA$RENAL_CHRONIC <- NULL

#particionar as partições
ind <- createDataPartition(FA$OBITO, p=1, times=1, list = F)
subdataset <- FA[ind, ]


ind <- createDataPartition(subdataset$OBITO, p=0.5, times=3, list = F)
treino <- subdataset[ind, ]
testando <- subdataset[-ind, ]
prop.table(table(treino$OBITO))
prop.table(table(testando$OBITO))



#--------------------------------------------------------------


rf <-randomForest(OBITO~.,data=subdataset, ntree=500) 
print(rf)

#floor(sqrt(ncol(subdataset) - 1))

mtry <- tuneRF(subdataset[,-13],subdataset$OBITO, ntreeTry=5000,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

#set.seed(71)
rf <-randomForest(OBITO~.,data=subdataset, mtry=best.m, importance=TRUE,ntree=5000)
print(rf)
#Evaluate variable importance
importance(rf)
varImpPlot(rf)

#--------------------------------------------------------------


#xteste argumento que passa a amostra de teste
#ytest variável de resposta
#ntree quantidade de árvores que irão compor a floresta
#mtree quantidade de variáveis aleatórias 
# bootstrap
#quant mínima de observações
aFloresta <- randomForest(x = treino[, -10],
                          y = treino$OBITO,
                          xtest = testando[, -10],
                          ytest = testando$OBITO,
                          ntree = 100,
                          mtree = 50,
                          replace = T,
                          nodesize = 100,
                          maxnodes = 100,
                          keep.forest = T
)

print(aFloresta)
importance(rf)
varImpPlot(aFloresta, main ="Relevancia")
plot(aFloresta,main ="Relevancia")

idadeAbaixo50 <- filter (FA, AGE <=50, OBITO=="1")

plot(idadeAbaixo50)
