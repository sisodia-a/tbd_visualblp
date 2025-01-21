library(dplyr)
exp_uk_product_data <- read.csv('exp_uk_product_data.csv',stringsAsFactors=FALSE)
exp_xi_fe <- read.csv('exp_opt_xi_fe.csv',stringsAsFactors=FALSE)
exp_python_image_table <- read.csv('exp_python_image_table.csv',stringsAsFactors=FALSE)

exp_uk_product_data <- cbind(exp_uk_product_data,exp_xi_fe)
exp_uk_product_data$xi_fe <- round(exp_uk_product_data$xi_fe,3)
temp <- exp_uk_product_data %>% select(clustering_ids,xi_fe) %>% distinct()

exp_python_image_table$xi_fe <- NULL
exp_python_image_table <- merge(exp_python_image_table,temp)

write.csv(exp_python_image_table,'exp_python_image_table.csv',row.names=FALSE)
