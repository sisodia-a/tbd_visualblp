## Christie's Data

rm(list=ls())

library(dplyr)
library(stringr)
library(ggplot2)
library(ggcorrplot)
library(cowplot)
library(stargazer)
# library(broom)
library(caret)
library(expm)

args <- commandArgs(trailingOnly = TRUE)
ARGUMENT <- args[1]

# beta <- c(1,5,10,20,30,40,50)
# m <- c(0,1,5,10,20,30,40,50)

beta <- 50
m <- 5

delete_columns <- function(df) {
for(i in ncol(df):1)
{
  mean <- mean(df[,i])
  sd <- sd(df[,i])
  if(mean==0 & sd==0)
  {
    df[,i] <- NULL
  }
}
  return(df)
}

calculate_udr <- function(z_i,z_j) {
  df <- matrix(NA,ncol(z_j),ncol(z_i))
  for(i in 1:ncol(z_i))
  {
    for(j in 1:ncol(z_j))
    {
      df[j,i] <- cor(z_i[,i], z_j[,j],method="spearman")
    }
  }
  df <- abs(df)
#  df <- sqrtm(t(df) %*% df)
  udr_row <- 0
  for(i in 1:nrow(df))
  {
    udr_row <- udr_row + (max(df[i,]) * max(df[i,]))/sum(df[i,])
  }
  udr_col <- 0
  for(j in 1:ncol(df))
  {
    udr_col <- udr_col + (max(df[,j]) * max(df[,j]))/sum(df[,j])
  }
  udr_score <- (udr_row+udr_col)/(nrow(df)+ncol(df))
  # print(paste("UDR Score", udr_score))
  return(udr_score)
}

udr_beta_inner <- function(beta,seed,m) {
  focal_df <- read.csv(paste0(ARGUMENT,"_s",seed,"b",beta,"m",m,"_total.csv"))
  focal_df$file_name <- NULL
  focal_df <- delete_columns(focal_df)
  udr_inner = 0
  for(i in 1:10)
  {
    if(i!=seed)
    {
      comparison_df <- read.csv(paste0(ARGUMENT,"_s",i,"b",beta,"m",m,"_total.csv"))
      comparison_df$file_name <- NULL
      comparison_df <- delete_columns(comparison_df)
      udr_inner <- udr_inner + calculate_udr(focal_df,comparison_df)
      }
  }
  # print(paste("Mean UDR Inner",udr_inner/9))
  return(udr_inner/9)
}

udr_beta_outer <- function(beta,m)
{
  udr_outer <- 0
  for(i in 1:10)
  {
    udr_outer <- udr_outer + udr_beta_inner(beta,i,m)
  }
  # print(paste("Mean UDR Outer",udr_outer/10))
  return((udr_outer)/10)
}

df_udr <- expand.grid(beta,m)
colnames(df_udr) <- c("beta","m")
df_udr$udr_values <- NA
for(i in 1:nrow(df_udr))
{
  df_udr$udr_values[i] <- udr_beta_outer(df_udr$beta[i],df_udr$m[i])
  print(i)
}

print(df_udr)

df_udr <- df_udr[order(df_udr$udr_values),]

df_udr[which.max(df_udr$udr_values),]

write.csv(df_udr,paste0(ARGUMENT,"_all_df_udr.csv"),row.names=FALSE)
