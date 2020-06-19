##########
#  Q2.3
##########

par(mfrow=c(1,1))


library(mvtnorm)
library(class)
library(e1071)

generate=function(n,d,mu,h){
  rmvnorm(n,mu,h^2*diag(d))
}

d=c(2,5,10,20,50,100,200,500) # dimensions

#mu1=rep(0,d[i])

#mu2=rep(0,d[i])
#mu2=c(1,rep(0,d[i]-1))
#mu2=rep(1,d[i])

x_train=list()
x_test=list()


bayes=function(x,mu1,mu2,d)dmvnorm(x,mean=mu1,sigma=diag(d))-dmvnorm(x,mean=mu2,sigma = 1/4*diag(d))

kde=function(x,x_sample,d,h,n){
  sum=0
  for(i in 1:n)sum=sum+dmvnorm(x,mean=x_sample[i,],sigma=(h^2)*diag(d))
  est=sum/n
}
###
# q2.3.3

MR_Bayes=rep(0,length(d))
MR_knn=rep(0,length(d))
MR_svm_1=rep(0,length(d))
MR_svm_2=rep(0,length(d))
MR_kde=rep(0,length(d))


set.seed(1234)

for(i in 1:length(d)){
  #params
  mu1=rep(0,d[i])
  mu2=rep(0,d[i])
  
  #================
  # data
  #================
  x11=generate(10,d[i],mu1,1)
  x12=generate(10,d[i],mu2,1/2)
  x_train[[i]]=rbind(x11,x12)
  y_train=c(rep(0,10),rep(1,10))
  
  x21=generate(100,d[i],mu1,1)
  x22=generate(100,d[i],mu2,1/2)
  x_test[[i]]=rbind(x21,x22)
  y_test=c(rep(0,100),rep(1,100)) 
  
  # Bayes
  y_pred=ifelse(apply(x_test[[i]],1,bayes,mu1,mu2,d[i])>0,0,1)
  MR_Bayes[i]=mean(y_test!=y_pred)
  
  # 1NN
  knn.pred=knn(x_train[[i]],x_test[[i]],y_train,k=1)
  MR_knn[i]=mean(knn.pred!=y_test)
  
  dat=data.frame(x_train[[i]])
  dat$y=as.factor(y_train)
  
  # SVM linear
  svmfit_1=svm(y~.,dat,kernel='linear',scale=F)
  y_pred=predict(svmfit_1,x_test[[i]])
  MR_svm_1[i]=mean(y_test!=y_pred)
  
  # SVM radial
  svmfit_2=svm(y~.,dat,kernel='radial',scale=F)
  y_pred=predict(svmfit_2,x_test[[i]])
  MR_svm_2[i]=mean(y_test!=y_pred)
  
  print(paste('computing KDA for d=',d[i]))
  
  # KDA
  y_pred=numeric(200)
  for(j in 1:200){
    dist1=as.matrix(dist(rbind(x11,x_test[[i]][j,])))
    h1=dist1[11,which.min(dist1[11,-11])]
    #print(h1)
    p0=kde(x_test[[i]][j,],x11,d=d[i],h=h1,n=10)
    
    dist2=as.matrix(dist(rbind(x12,x_test[[i]][j,])))
    h2=dist2[11,which.min(dist2[11,-11])]
    #print(h2)
    p1=kde(x_test[[i]][j,],x12,d=d[i],h=h2,n=10)
    
    y_pred[j]=ifelse(p0<p1,1,0)
  }
  
  MR_kde[i]=mean(y_test!=y_pred)
  
  
}


# test error plots
MR=matrix(c(MR_Bayes,MR_knn,MR_svm_1,MR_svm_2,MR_kde),ncol=length(d),byrow=T)
names=c('Bayes','1NN','SVM_linear','SVM_radial','KDE')
rownames(MR)=names
matplot(t(MR),type='b',lwd=1,pch=20,ylab='Error rate',xlab='Dimensions',col=1:5,lty=1,xaxt='n')
axis(at=1:length(d),side=1,las=0,labels = d)
legend('bottomleft',inset=0.01,col=c(1:5),names,pch=20)

###
# q2.3.1
MR_Bayes=rep(0,length(d))
MR_knn=rep(0,length(d))
MR_svm_1=rep(0,length(d))
MR_svm_2=rep(0,length(d))
MR_kde=rep(0,length(d))


set.seed(1234)

for(i in 1:length(d)){
  #params
  mu1=rep(0,d[i])
  mu2=c(1,rep(0,d[i]-1))
  
  #================
  # data
  #================
  x11=generate(10,d[i],mu1,1)
  x12=generate(10,d[i],mu2,1/2)
  x_train[[i]]=rbind(x11,x12)
  y_train=c(rep(0,10),rep(1,10))
  
  x21=generate(100,d[i],mu1,1)
  x22=generate(100,d[i],mu2,1/2)
  x_test[[i]]=rbind(x21,x22)
  y_test=c(rep(0,100),rep(1,100)) 
  
  # Bayes
  y_pred=ifelse(apply(x_test[[i]],1,bayes,mu1,mu2,d[i])>0,0,1)
  MR_Bayes[i]=mean(y_test!=y_pred)
  
  # 1NN
  knn.pred=knn(x_train[[i]],x_test[[i]],y_train,k=1)
  MR_knn[i]=mean(knn.pred!=y_test)
  
  dat=data.frame(x_train[[i]])
  dat$y=as.factor(y_train)
  
  # SVM linear
  svmfit_1=svm(y~.,dat,kernel='linear',scale=F)
  y_pred=predict(svmfit_1,x_test[[i]])
  MR_svm_1[i]=mean(y_test!=y_pred)
  
  # SVM radial
  svmfit_2=svm(y~.,dat,kernel='radial',scale=F)
  y_pred=predict(svmfit_2,x_test[[i]])
  MR_svm_2[i]=mean(y_test!=y_pred)
  
  print(paste('computing KDA for d=',d[i]))
  # KDA
  y_pred=numeric(200)
  for(j in 1:200){
    dist1=as.matrix(dist(rbind(x11,x_test[[i]][j,])))
    h1=dist1[11,which.min(dist1[11,-11])]
    #print(h1)
    p0=kde(x_test[[i]][j,],x11,d=d[i],h=h1,n=10)
    
    dist2=as.matrix(dist(rbind(x12,x_test[[i]][j,])))
    h2=dist2[11,which.min(dist2[11,-11])]
    #print(h2)
    p1=kde(x_test[[i]][j,],x12,d=d[i],h=h2,n=10)
    
    y_pred[j]=ifelse(p0<p1,1,0)
  }
  
  MR_kde[i]=mean(y_test!=y_pred)
  
  
}


# test error plots
MR=matrix(c(MR_Bayes,MR_knn,MR_svm_1,MR_svm_2,MR_kde),ncol=length(d),byrow=T)
names=c('Bayes','1NN','SVM_linear','SVM_radial','KDE')
rownames(MR)=names
matplot(t(MR),type='b',lwd=1,pch=20,ylab='Error rate',xlab='Dimensions',col=1:5,lty=1,xaxt='n')
axis(at=1:length(d),side=1,las=0,labels = d)
legend('bottomleft',inset=0.01,col=c(1:5),names,pch=20)

###
# q2.3.2

MR_Bayes=rep(0,length(d))
MR_knn=rep(0,length(d))
MR_svm_1=rep(0,length(d))
MR_svm_2=rep(0,length(d))
MR_kde=rep(0,length(d))


set.seed(1234)

for(i in 1:length(d)){
  #params
  mu1=rep(0,d[i])
  mu2=rep(1,d[i])
  
  #================
  # data
  #================
  x11=generate(10,d[i],mu1,1)
  x12=generate(10,d[i],mu2,1/2)
  x_train[[i]]=rbind(x11,x12)
  y_train=c(rep(0,10),rep(1,10))
  
  x21=generate(100,d[i],mu1,1)
  x22=generate(100,d[i],mu2,1/2)
  x_test[[i]]=rbind(x21,x22)
  y_test=c(rep(0,100),rep(1,100)) 
  
  # Bayes
  y_pred=ifelse(apply(x_test[[i]],1,bayes,mu1,mu2,d[i])>0,0,1)
  MR_Bayes[i]=mean(y_test!=y_pred)
  
  # 1NN
  knn.pred=knn(x_train[[i]],x_test[[i]],y_train,k=1)
  MR_knn[i]=mean(knn.pred!=y_test)
  
  dat=data.frame(x_train[[i]])
  dat$y=as.factor(y_train)
  
  # SVM linear
  svmfit_1=svm(y~.,dat,kernel='linear',scale=F)
  y_pred=predict(svmfit_1,x_test[[i]])
  MR_svm_1[i]=mean(y_test!=y_pred)
  
  # SVM radial
  svmfit_2=svm(y~.,dat,kernel='radial',scale=F)
  y_pred=predict(svmfit_2,x_test[[i]])
  MR_svm_2[i]=mean(y_test!=y_pred)
  
  print(paste('computing KDA for d=',d[i]))
  # KDA
  y_pred=numeric(200)
  for(j in 1:200){
    dist1=as.matrix(dist(rbind(x11,x_test[[i]][j,])))
    h1=dist1[11,which.min(dist1[11,-11])]
    #print(h1)
    p0=kde(x_test[[i]][j,],x11,d=d[i],h=h1,n=10)
    
    dist2=as.matrix(dist(rbind(x12,x_test[[i]][j,])))
    h2=dist2[11,which.min(dist2[11,-11])]
    #print(h2)
    p1=kde(x_test[[i]][j,],x12,d=d[i],h=h2,n=10)
    
    y_pred[j]=ifelse(p0<p1,1,0)
  }
  
  MR_kde[i]=mean(y_test!=y_pred)
  
  
}


# test error plots
MR=matrix(c(MR_Bayes,MR_knn,MR_svm_1,MR_svm_2,MR_kde),ncol=length(d),byrow=T)
names=c('Bayes','1NN','SVM_linear','SVM_radial','KDE')
rownames(MR)=names
matplot(t(MR),type='b',lwd=1,pch=20,ylab='Error rate',xlab='Dimensions',col=1:5,lty=1,xaxt='n')
axis(at=1:length(d),side=1,las=0,labels = d)
legend('bottomleft',inset=0.01,col=c(1:5),names,pch=20)

