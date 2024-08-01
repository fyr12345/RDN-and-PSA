#Create a function that performs hierarchical clustering and returns the highest cluster index
cluster_test <- function(cca_data){
  diss<- dist(cca_data, method = "cosine")
  hcfit <- NbClust(diss =diss,cca_data,  distance=NULL ,method="mcquitty", index="ch", min.nc=2, max.nc = 10)
  CH_index <- max(hcfit$All.index)
  return(c("CH"=CH_index))
}
library(ggplot2)
library(proxy)
path="XX.xlsx"##Storing brain scores of different dimensions for each patient
data=readxl::read_xlsx(path,sheet = 1)
data<-data[,c(1,2)]##Selection of salient dimensions

####real data
disss<- dist(data, method = "cosine")
hcfitttt <- NbClust(diss =disss,data,  distance=NULL ,method="mcquitty", index="ch", min.nc=2, max.nc = 10)
hcfitttt <- cluster_test(data)


#Fitting a multivariate normal distribution to the same data used to perform hierarchical clustering
library(NbClust)
library(MASS)
real_CI <- cluster_test(data)
sigma <- cov(data)
mu <- colMeans(data)

#Get an null cluster distribution
null_CI <- list()
n_sims <- 1999
for (i in 1:n_sims){
  rand_sample <- mvrnorm(n=nrow(data), mu=mu, Sigma=sigma)
  null_CI[[i]] <- cluster_test(rand_sample)
}
null_CI <- as.data.frame(do.call(rbind, null_CI))

rank_cv1 <- sum(real_CI[1] < null_CI[,1]) + 1
pval_cv1 <- rank_cv1 / (n_sims+1)

t((c("p.val variance ratio"=pval_cv1)))

