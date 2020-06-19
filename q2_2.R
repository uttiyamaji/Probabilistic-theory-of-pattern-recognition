set.seed(1234)

generate=function(n,d){
  mu=numeric(d)
  sigma=diag(d)
  x=rmvnorm(n,mu,sigma)
}


d=c(2,5,10,20,50,100,200,500)


angle=function(x,y){
  s=sum(x*y)/(sqrt(sum(x^2))*sqrt(sum(y^2)))
  return(acos(s))
}

a=list()

for(i in 1:length(d)){
  x_sample=generate(3,d[i])
  a[[i]]=matrix(0,nrow=3,ncol=3)
  for(j in 1:3){
    for(k in j:3){
      if(j!=k)
        a[[i]][j,k]=angle(x_sample[j,],x_sample[k,])
    }
  }
}
angles=matrix(0,ncol=3,nrow=length(d))
for(i in 1:length(d)){
  angles[i,]=a[[i]][c(4,7,8)]
}

colnames(angles)=c('X1','X2','X3')
rownames(angles)=d

angles

