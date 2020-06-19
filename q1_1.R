############################################
#			Q1
############################################

###########
# Q1.1
###########

library(mvtnorm)
library(e1071)
library(class)
library(MASS)

# function to generate d-dim Normal(mu,I)

generate=function(n,d,mu){
  rmvnorm(n,mu,diag(d))
}

# to estimate kernel density using gaussian kernels
kde=function(x,x_sample,d,h,n){
  sum=0
  for(i in 1:n)sum=sum+dmvnorm(x,mean=x_sample[i,],sigma=(h^2)*diag(d))
  est=sum/n
}

# estimates kernel density with uniform kernel
kde_uniform=function(x,x_sample,d,h,n){
  sum=0
  for(i in 1:n)sum=sum+ifelse(abs(x-x_sample[i,])<h/2,1,0)
  est=sum/n
}

# to calculate bayes risk for this particular problem

bayes=function(x,mu1,mu2,d,pi0){
  pi0*dmvnorm(x,mean=mu1,sigma=diag(d))-(1-pi0)*dmvnorm(x,mean=mu2,sigma =diag(d))
}

d=c(2,10,25,50) # dimensions 

x_train=list();x_test=list() # contains all the data

# placeholders for Misclassification rates for the classifiers

MR_Bayes_test=rep(0,length(d))
MR_knn_test=rep(0,length(d))
MR_svm_1_test=rep(0,length(d))
MR_svm_2_test=rep(0,length(d))
MR_kde_test=rep(0,length(d))
MR_log_test=rep(0,length(d))
MR_lda_test=rep(0,length(d))
MR_qda_test=rep(0,length(d))
MR_lm_test=rep(0,length(d))
MR_kde_unif_test=rep(0,length(d))

K=rep(0,length(d)) # contains all the choices of K in kNN for each dimension

MR_Bayes_train=rep(0,length(d))
MR_knn_train=rep(0,length(d))
MR_svm_1_train=rep(0,length(d))
MR_svm_2_train=rep(0,length(d))
MR_kde_train=rep(0,length(d))
MR_log_train=rep(0,length(d))
MR_lda_train=rep(0,length(d))
MR_qda_train=rep(0,length(d))
MR_lm_train=rep(0,length(d))
MR_kde_unif_train=rep(0,length(d))

set.seed(42)

options(warn=-1)

for(i in 1:length(d)){
  
  # params for data
  mu1=rep(0,d[i])
  mu2=c(d[i],rep(0,(d[i]-1)))
  
  #==================
  # data generation
  #==================
  x11=generate(100,d[i],mu1)
  x12=generate(100,d[i],mu2)
  x_train[[i]]=rbind(x11,x12)
  y_train=c(rep(0,100),rep(1,100))
  
  x21=generate(250,d[i],mu1)
  x22=generate(250,d[i],mu2)
  x_test[[i]]=rbind(x21,x22)
  y_test=c(rep(0,250),rep(1,250))
  
  print(paste('data generated for d=',d[i]))
  
  #==============
  # Bayes error 
  #==============
  
  y_pred=ifelse(apply(x_test[[i]],1,bayes,mu1,mu2,d[i],0.5)>0,0,1)
  MR_Bayes_test[i]=mean(y_test!=y_pred)
  
  y_pred_train=ifelse(apply(x_train[[i]],1,bayes,mu1,mu2,d[i],0.5)>0,0,1)
  MR_Bayes_train[i]=mean(y_train!=y_pred_train)
  
  print('Bayes risk calculated')
  
  
  dat_train=data.frame(x_train[[i]])
  dat_train$y=y_train
  dat_test=data.frame(x_test[[i]])
  #dat_test$y=as.factor(y_test)
  
  #========
  # LDA
  #========
  lda.fit=lda(y~.,dat_train)
  
  # predicting on test set
  pred=predict(lda.fit,dat_test)
  MR_lda_test[i]=mean(pred$class!=y_test)
  
  # predicting on train set
  pred_train=predict(lda.fit,dat_train)
  MR_lda_train[i]=mean(pred_train$class!=y_train)
  
  #======
  # QDA
  #======
  qda.fit=qda(y~.,dat_train)
  
  pred=predict(qda.fit,dat_test)
  MR_qda_test[i]=mean(pred$class!=y_test)
  
  pred_train=predict(qda.fit,dat_train)
  MR_qda_train[i]=mean(pred_train$class!=y_train)
  
  print('LDA QDA done')
  
  # Logistic Regression
  
  glm.fit=glm(as.factor(y)~.,dat_train,family='binomial')
  probs=predict(glm.fit,dat_test,type='response')
  #print(probs)
  pred=rep(0,500)
  pred[probs>0.5]=1
  pred[is.na(pred)==T]=0
  MR_log_test[i]=mean(y_test!=pred)
  
  print('Logistic done')
  
  #==========
  # SVM
  #==========
  
  # SVM Linear
  svmfit_1=svm(as.factor(y)~.,dat_train,kernel='linear',scale=F)
  y_pred=predict(svmfit_1,x_test[[i]])
  MR_svm_1_test[i]=mean(y_test!=y_pred)
  
  y_pred_train=predict(svmfit_1,x_train[[i]])
  MR_svm_1_train[i]=mean(y_train!=y_pred_train)
  
  # SVM radial
  svmfit_2=svm(as.factor(y)~.,dat_train,kernel='radial',scale=F)
  y_pred=predict(svmfit_2,x_test[[i]])
  MR_svm_2_test[i]=mean(y_test!=y_pred)
  
  y_pred_train=predict(svmfit_2,x_train[[i]])
  MR_svm_2_train[i]=mean(y_train!=y_pred_train)
  
  print('SVM done')
  
  #==============
  # kNN
  #==============
  error=numeric(10)
  for(k in 1:10){
    sum=0
    for(j in 1:200){
      train=x_train[[i]][-j,]
      test=x_train[[i]][j,]
      knn.pred=knn(train,test,y_train[-j],k=k)
      sum=sum+sum(knn.pred!=y_train[j])
    }
    error[k]=sum/200
  }
  K[i]=which.min(error)
  knn.pred=knn(x_train[[i]],x_test[[i]],y_train,k=K[i])
  MR_knn_test[i]=mean(knn.pred!=y_test)
  
  print('kNN done')
  
  print('KDE started...will take time')
  
  #==================
  #  KDE
  #================== 
  y_pred=numeric(500)
  for(j in 1:500){
    dist1=as.matrix(dist(rbind(x11,x_test[[i]][j,])))  # calculating the distance from all the obs
    h1=dist1[101,which.min(dist1[101,-101])]  # taking h as the dist from nearest neighbour
    #print(h1)
    p0=kde(x_test[[i]][j,],x11,d=d[i],h=h1,n=100)
    
    dist2=as.matrix(dist(rbind(x12,x_test[[i]][j,])))
    h2=dist2[101,which.min(dist2[101,-101])]
    #print(h2)
    p1=kde(x_test[[i]][j,],x12,d=d[i],h=h2,n=100)
    
    y_pred[j]=ifelse(p0<p1,1,0)
  }
  #print(mean(y_pred))
  MR_kde_test[i]=mean(y_test!=y_pred)
  
  y_pred=numeric(500)
  for(j in 1:500){
    dist1=as.matrix(dist(rbind(x11,x_test[[i]][j,])))
    h1=dist1[101,which.min(dist1[101,-101])]
    #print(h1)
    p0=kde_uniform(x_test[[i]][j,],x11,d=d[i],h=h1,n=100)
    
    dist2=as.matrix(dist(rbind(x12,x_test[[i]][j,])))
    h2=dist2[101,which.min(dist2[101,-101])]
    #print(h2)
    p1=kde_uniform(x_test[[i]][j,],x12,d=d[i],h=h2,n=100)
    
    y_pred[j]=ifelse(p0<p1,1,0)
  }
  MR_kde_unif_test[i]=mean(y_test!=y_pred)
  print('KDE done')
  
  
  
  # Linear regression
  dat_train$y[y_train==0]=-1
  lm.fit=lm(as.numeric(y)~.,dat_train)
  pred=predict(lm.fit,dat_test)
  pred=ifelse(pred>0,1,0)
  #table(y_test,pred)
  #pred=as.factor(pred)
  MR_lm_test[i]=mean(y_test!=pred)
  
  pred_train=predict(lm.fit,dat_train)
  pred_train=ifelse(pred_train>0,1,0)
  MR_lm_train[i]=mean(y_train!=pred_train)
  
  print('lm done')
} 

#===========================
# to plot the test errors
#===========================
MR_test=matrix(c(MR_Bayes_test,MR_knn_test,MR_svm_1_test,MR_svm_2_test,MR_kde_test,MR_kde_unif_test,MR_lda_test,MR_qda_test,MR_lm_test,MR_log_test),ncol=length(d),byrow=T)
names=c('Bayes','kNN','SVM_linear','SVM_radial','KDA1','KDA2','LDA','QDA','lm','logis')
rownames(MR_test)=names
matplot(t(MR_test),type='b',lwd=1,pch=20,ylab='Error rate',xlab='Dimensions',col=1:10,lty=1,xaxt='n',main='Test errors')
legend('topright',inset=0.01,col=1:10,names,pch=20)
axis(at=1:length(d),side=1,las=0,labels = d)

#==========================
# to plot the train  errors
#==========================
MR_train=matrix(c(MR_Bayes_train,MR_lda_train,MR_qda_train,MR_svm_1_train,MR_svm_2_train,MR_lm_train),ncol=length(d),byrow=T)
matplot(t(MR_train),type='b',lwd=1,pch=20,ylab='Error rate',xlab='Dimensions',col=1:9,lty=1,xaxt='n',main='train errors')
names=c('Bayes','LDA','QDA','SVM_linear','SVM_radial','lm')
axis(at=1:length(d),side=1,las=0,labels = d)
legend('topright',inset=0.01,col=1:9,names,pch=20)


# LDA plots
plot(MR_lda_test,type='b',ylim=c(0,1),xlab='Dimensions',xaxt='n',ylab='error',main='LDA')
lines(1:4,MR_lda_train,type='b',lty=2)
axis(at=1:length(d),side=1,las=0,labels = d)

# QDA plots
plot(MR_qda_test,type='b',ylim=c(0,1),xlab='Dimensions',xaxt='n',ylab='error',main='QDA')
lines(1:4,MR_qda_train,type='b',lty=2)
axis(at=1:length(d),side=1,las=0,labels = d)

# Bayes plots
plot(MR_Bayes_test,type='b',ylim=c(0,1),xlab='Dimensions',xaxt='n',ylab='error',main='Bayes')
lines(1:4,MR_Bayes_train,type='b',lty=2)
axis(at=1:length(d),side=1,las=0,labels = d)

# SVM Linear plots
plot(MR_svm_1_test,type='b',ylim=c(0,1),xlab='Dimensions',xaxt='n',ylab='error',main='SVM Linear')
lines(1:4,MR_svm_1_train,type='b',lty=2)
axis(at=1:length(d),side=1,las=0,labels = d)

# SVM radial plots
plot(MR_svm_2_test,type='b',ylim=c(0,1),xlab='Dimensions',xaxt='n',ylab='error',main='SVM radial')
lines(1:4,MR_svm_2_train,type='b',lty=2)
axis(at=1:length(d),side=1,las=0,labels = d)
