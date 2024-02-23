#preparacao do ambiente
rm(list = ls())
cat("\014")  # clear console
dir_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir_path)

#usada para gerar matrix de confusao
library(caret)

#CARREGA A FUNCAO DEGRAU BIPOLAR QUE SERA UTILIZADA NO NEURONIO
#funcao degrau bipolar
degrauBipolar <- function(u)
{
  if (u>=0)
    y <- 1
  else
    y <- -1
  
  return(y)
}

#CARREGA A FUNCAO QUE CLASSIFICARA A AMOSTRA COM BASE NA REDE JA TREINADA
previsao <- function(w,elemento)
{
  u <- t(w) %*% as.numeric(elemento)
  yhat <- degrauBipolar(u)
  
  return(yhat)
}

#----------------------------------------------------------
#ATIVIDADE 3

#1 - UTILIZE A REDE PERCEPTRON COM APRENDIZADO POR REGRA DE HEBB PARA CLASSIFICAR DOIS TIPOS DE FLORES COM BASE EM 4 CARACTERISTICAS CONFORME ARQUIVO iris.csv

#Sao 50 amostras de Setosa e 50 de versicolor

#O dataset iris deve ser carregado da seguinte forma:

dados <- read.table("iris.csv",header=T,sep=",")
dados$variety <- ifelse(dados$variety %in% "Setosa", -1, dados$variety)
dados$variety <- ifelse(dados$variety %in% "Versicolor", 1, dados$variety)

setosa <- dados[dados$variety==-1,]
versicolor <- dados [dados$variety==1,]

setosaMistura<- setosa[sample(nrow(setosa)),]

versicolorMistura<- versicolor[sample(nrow(versicolor)),]
#-------------------- nº 2 separando conjunto de teste e conjunto de validação --------------------
setosaT<- setosaMistura[1:30,]
versicolorT<- versicolorMistura [1:30,]
setosaR<- setosaMistura[31:50,]
versicolorR<- versicolorMistura[31:50,]
treino <- rbind(setosaT, versicolorT)
teste<- rbind(setosaR, versicolorR)
#deletando
rm(setosaMistura)
rm(setosaR)
rm(setosaT)
rm(versicolorMistura)
rm(versicolorR)
rm(versicolorT)

#--------------------Nº 3 treinando a RNA-------------------- 
#COLETANDO INFORMACOES DO DATASET
#quantidade de elementos na amostra
N <- dim(treino)[1]
#quantidade de entradas (a subtracao de 1 diz respeito ao fato de que a ultima
#coluna corresponde a saida esperada e nao a um atributo/entrada)
n <- dim(treino)[2] - 1
#separando a amostra de entrada
amostra <- treino[,1:n]
#inserindo o bias
bias <- rep(-1,N)
entradas <- cbind(amostra,bias)

#capturando apenas a saida y esperada
y <- treino[,n+1]

#removendo variaveis nao mais utilizadas da memoria
rm(bias)
rm(dados)
rm(amostra)

#PASSO 3: Iniciar o vetor w com valores aleatorios pequenos
#gerando o vetor de pesos iniciais aleatoriamente
#o +1 e referente ao peso w0 do bias
w <- runif(n+1, min=-0.3, max=0.3)

#PASSO 4: Especificar a taxa de aprendizagem eta
eta <- 0.07

#PASSO 5: Iniciar o contador de numero de epocas
nepocas <- 0

#COLOCANDO ALGUNS CRITERIOS DE PARADA
#Definir o maximo de epocas permitido
maxepocas <- 300
#Tolerancia minima aceitavel
tol <- 0

#VARIAVEL QUE IRA DENOTAR A EXISTENCIA DE ERRO NA EPOCA CORRENTE
#Por exemplo, abaixo esta sendo setado do valor dela pois sabemos
#que na primeira iteracao existe erro
eepoca <- tol+1

#VARIAVEL QUE VAI ARMAZENAR O ERRO TOTAL COMPUTADO EM CADA EPOCA
erromedioEpocaAEpoca<-matrix(nrow=1,ncol=maxepocas)

#PASSO 6: Repetir instrucoes
while ((nepocas<maxepocas) && (eepoca>tol))
{
  #variavel que vai armazenar o erro total a
  #cada apresentacao do conjunto de dados
  erroConvergencia <- 0
  
  #distribui os indices dos elementos da amostra de forma aleatoria
  #para retirar o determinismo do processo, ou seja, embaralha os
  #elementos e nao apresenta eles ordenadamente a RNA
  indElementosEmbaralhados <- sample(N)
  
  for (i in 1:N)
  {
    #captura o indice do elemento a ser apresentado a rede
    indiceCorrente <- indElementosEmbaralhados[i]
    
    #captura o elemento corrente
    elemento <- as.numeric(entradas[indiceCorrente,])
    
    #calcula o limiar de ativacao
    u <- t(w) %*% elemento
    
    #gera a saida prevista com base na funcao de ativacao
    yhat <- degrauBipolar(u)
    
    #calcula o erro
    erro <-as.numeric(y[indiceCorrente])-yhat
    
    #calculo do delta que incrementara ou decrementara os pesos
    delta<-(eta*erro)*elemento
    
    #substitui os pesos antigos pelos novos pesos apos esta epoca
    w <- w + delta
    
    #somatorio dos erros de convergencia epoca a epoca
    #o erro esta elevado ao quadrado pois pode ser que o mesmo seja
    #negativo. Estamos interessados na distancia para o ponto desejado
    erroConvergencia <- erroConvergencia + (erro^2)
  }
  
  #incrementa o contador de epocas para saber em que epoca a RNA esta
  nepocas <- nepocas+1
  
  #calcula a media dos erros na epoca corrente 
  #e armazena o resultado num vetor para que, depois,
  #seja possivel avaliar o erro caindo epoca apos epoca
  erromedioEpocaAEpoca[nepocas] <- erroConvergencia/N
  
  #a variavel eepoca e utilizada no laco do while para saber
  #se o erro medio existente ate o momento e maior ou menor do que
  #a tolerancia estabelecida. E um criterio de parada.
  eepoca <- erromedioEpocaAEpoca[nepocas]
}


#plotando a convergencia do algoritmo com base no erro epoca a epoca
plot(as.numeric(erromedioEpocaAEpoca), type="l", ylab="erro medio", xlab="epocas", main="Convergencia do erro")

#exibindo os pesos calibrados pela regra de Hebb
print("Conjunto ideal de pesos:")
print(w)

#exibindo epocas necessarias a convergencia
print("Epocas necessarias a convergencia:")
print(nepocas)

#-------------------- 4 aplicando a RNA --------------------
#COLETANDO INFORMACOES DO DATASET
#quantidade de elementos na amostra
N <- dim(teste)[1]
#quantidade de entradas (a subtracao de 1 diz respeito ao fato de que a ultima
#coluna corresponde a saida esperada e nao a um atributo/entrada)
n <- dim(teste)[2] - 1
#separando a amostra de entrada
amostra2 <- teste[,1:n]
#inserindo o bias
bias <- rep(-1,N)
entradas2 <- cbind(amostra2,bias)
#capturando apenas a saida y esperada
y2 <- teste[,n+1]

#ALGORITMO DE CLASSIFICACAO
#cria um vetor para receber todas as previsoes realizadas pela rede
yhat2 <- rep(NA,N)
#armazena todas as previsoes. uma para cada elemento da amostra.
for (i in 1:N)
{
  yhat2[i] <- previsao(w,entradas2[i,])
}

#converte o -1 em 0 para uso da funcao que gera a matrix de confusao
yhat2[which(yhat2==-1)] <- 0
y2[which(y2==-1)] <- 0

#gera a matrix de confusao
confusao <- confusionMatrix(table(y2,yhat2))
print(confusao)

#acerta os dados para geracao de um grafico de barras
resultado <- c(confusao[2]$table[1],
               confusao[2]$table[2],
               confusao[2]$table[4],
               confusao[2]$table[3])
barplot(resultado, names.arg = c("0-0","0-1","1-1","1-0"),col = c("green","red","green","red"),xlab="Previsoes",ylab="Quantidade")


#5----------------------

#alterando a tolerancia para 1 e a taxa para 0.08
#alterando a tolerancia para 2 e a taxa para 0.09



