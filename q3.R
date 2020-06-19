set.seed(1234)

x=runif(100,-2,2)
e=rnorm(100,0,1)

y=2*x^3+x^2-2*x+5+e

plot(x,y,cex=1,col='darkgrey',pch=20)

x.grid=seq(-2,2,0.02)

# plotting with regression fits for q=2,5,10,20
k=1
for(i in c(2,5,10,20)){
  mod=glm(y~poly(x,degree=i,raw=T))
  preds=predict(mod,newdata=list(x=x.grid))
  lines(x.grid,preds,lwd=0.5,col=k);k=k+1
}
names=c('q=2','q=5','q=10','q=20')
legend('topleft',inset=0.01,col=c(1:5),names,pch=20)

#===========
# AIC BIC
#========
mse=numeric(4)
aic=numeric(4)
bic=numeric(4)

q.grid=c(2,5,10,20)
for(i in 1:length(q.grid)){
  mod=glm(y~poly(x,degree=q.grid[i],raw=T))
  preds=predict(mod,newdata=list(x=x))
  mse[i]=sqrt(sum((preds-y)^2))
  aic[i]=AIC(mod)
  bic[i]=BIC(mod)
}

# plot the AIC, BIC 
par(mfrow=c(1,2))
plot(aic,type='b',pch=20,main='AIC',xaxt='n')
axis(at=c(1,2,3,4),side=1,labels=c(2,5,10,20))
abline(v=which.min(aic),col='red',lty=2)

plot(bic,type='b',pch=20,main='BIC',xaxt='n')
axis(at=c(1,2,3,4),side=1,labels=c(2,5,10,20))
abline(v=which.min(bic),col='red',lty=2)

par(mfrow=c(1,1))
# MSE
plot(mse,type='b',pch=20,main='MSE for different values of q',xaxt='n',xlab='q')
axis(at=c(1,2,3,4),side=1,labels=c(2,5,10,20))
abline(v=which.min(mse),col='red',lty=2)

#===================
# cross validation
#===================
dat=data.frame(x,y)

library(boot)
set.seed(1234)

q=c(2,5,10,20)

# loocv
cv.err.1=rep(0,4)
for (i in 1:length(q)) {
  glm.fit=glm(y~poly(x,q[i],raw=T),data=dat)
  cv.err.1[i]=cv.glm(dat,glm.fit)$delta[1]
}

# 2-fold
cv.err.2=rep(0,4)
for (i in 1:length(q)) {
  glm.fit=glm(y~poly(x,q[i],raw=T),data=dat)
  cv.err.2[i]=cv.glm(dat,glm.fit,K=2)$delta[1]
}

# 5-fold
cv.err.5=rep(0,4)
for (i in 1:length(q)) {
  glm.fit=glm(y~poly(x,q[i],raw=T),data=dat)
  cv.err.5[i]=cv.glm(dat,glm.fit,K=5)$delta[1]
}

cv=matrix(c(cv.err.1,cv.err.2,cv.err.5),ncol=4,nrow=3)
colnames(cv)=c('q=2','q=5','q=10','q=20')
rownames(cv)=c('1-fold','2-fold','5-fold')
cv
