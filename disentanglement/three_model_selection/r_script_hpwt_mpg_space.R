library(dplyr)
set.seed(1)

args <- commandArgs(trailingOnly = TRUE)
ARGUMENT <- args[1]

s <- c(1,2,3,4,5,6,7,8,9,10)
b <- 50
m <- 20
df <- expand.grid(s,b,m)
colnames(df) <- c("s","b","m")
df$train_losses_file <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_train_losses.csv")
df$mean_params_train <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_mean_params_train.csv")
df$mean_params_validation <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_mean_params_validation.csv")
df$filename_train <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_filename_train.csv")
df$filename_validation <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_filename_validation.csv")

for(i in 1:nrow(df))
{
mean_params_train <- read.csv(df$mean_params_train[i],stringsAsFactors = FALSE, header=FALSE)
filename_train <- read.csv(df$filename_train[i],stringsAsFactors = FALSE, header=FALSE)
colnames(filename_train) <- "file_name"
train <- cbind(filename_train,mean_params_train)
mean_params_validation <- read.csv(df$mean_params_validation[i],stringsAsFactors = FALSE, header=FALSE)
filename_validation <- read.csv(df$filename_validation[i],stringsAsFactors = FALSE, header=FALSE)
colnames(filename_validation) <- "file_name"
validation <- cbind(filename_validation,mean_params_validation)
total <- rbind(train,validation) %>% arrange(file_name)
print(dim(total))
write.csv(total,paste0(ARGUMENT,"_s",df$s[i],"b",df$b[i],"m",df$m[i],"_total.csv"),row.names=FALSE)
}

df$total <- paste0(ARGUMENT,"_s",df$s,"b",df$b,"m",df$m,"_total.csv")
for(i in 1:nrow(df))
{
total <- read.csv(df$total[i],stringsAsFactors = FALSE)
loss_file <- read.csv(df$train_losses_file[i],stringsAsFactors = FALSE) %>% filter(Epoch==199) %>% filter(grepl("kl_loss_training_",Loss))
loss_file$serial_no <- seq(1,20,1)
loss_file <- loss_file %>% filter(Value>0.10)
select_columns <- loss_file$serial_no + 1
total <- total[,c(1,select_columns)]
write.csv(total,paste0(ARGUMENT,"_s",df$s[i],"b",df$b[i],"m",df$m[i],"_total.csv"),row.names=FALSE) 
}

