if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EBImage")

library(EBImage)
library(keras)

##read images
pics<-c('p1.jpg','p2.jpg','p3.jpg','p4.jpg','p5.jpg','p6.jpg',
        'c1.jpg','c2.jpg','c3.jpg','c4.jpg','c5.jpg','c6.jpg')

mypics<-list()
for (i in 1:12) {mypics[[i]]<-readImage(pics[i])}


###explore

print(mypics)
print(mypics[[1]])
display(mypics[[1]])
summary(mypics[[1]])
hist(mypics[[1]])
str(mypics)



###resize


for (i in 1:12) {mypics[[i]]<-resize(mypics[[i]],28,28)}


##reshape

for (i in 1:12) {mypics[[i]]<-array_reshape(mypics[[i]],c(28,28,3))}



###row bind

trainx<-NULL

for (i in 1:5) {trainx<- rbind(trainx,mypics[[i]])}
for (i in 7:11) {trainx<- rbind(trainx,mypics[[i]])}
str(trainx)

testx<-rbind(mypics[[6]],mypics[[12]])
trainy<-c(1,1,1,1,1,0,0,0,0,0)
testy<-c(1,0)


###one hot encoding

trainlabels<-to_categorical(trainy)
testlabels<-to_categorical(testy)


###model

model<-keras_model_sequential()

model %>% 
  layer_dense(units = 256,activation = 'relu',input_shape = c(2352)) %>% 
  layer_dense(units = 128,activation = 'relu') %>% 
  layer_dense(units = 2,activation = 'softmax')


summary(model)



###compile

model %>% 
  compile(loss='binary_crossentropy'
          optimizer= 'optimizer_rmsprop()',
          metrics=c('accuracy'))



##fit model


history<-model %>% 
  fit(trainx,
      trainlabels,
      epochs=30,
      batch_size=32,
      validation_split=0.2)

plot(history)



### evaluation and prediction-   train data


model %>% evaluate(trainx,trainlabels)
pred<-model %>% predict_classes(trainx)

table(predicted=pred,actual=trainy)


prob<-model %>% predict_proba(trainx)

cbind(prob,predicted=pred,actual=trainy)




### evaluation and prediction-   test data


model %>% evaluate(testx,testlabels)
pred<-model %>% predict_classes(testx)

table(predicted=pred,actual=testy)

