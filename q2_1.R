#########
#  Q2.1
#########

set.seed(1234)
library(mvtnorm)

generate=function(n,d){
  mu=numeric(d)
  sigma=diag(d)
  x=rmvnorm(n,mu,sigma)
}


d=c(2,5,10,20,50,100,200,500)

length=matrix(0,ncol=length(d),nrow=20)

for(i in 1:length(d)){
  x_sample=generate(20,d[i])
  length[,i]=apply(x_sample,1,function(x)sqrt(sum(x^2)))
}
x=0:30
par(mfrow=c(4,2))
for(i in 1:8)plot(density(length[,i]),main=paste('d=',d[i]))
