require(caret)
library(pbapply)
require(drat)
require(mxnet)
setwd("/home/stathis/Desktop/itdrr2020/cnn/")
image.dir <- "//home/stathis/Desktop/itdrr2020/cnn/train/"
require(EBImage)
source("/home/stathis/Desktop/itdrr2020/cnn/extract_feature.R")
image.test <- readImage(file.path("/home/stathis/Desktop/itdrr2020/cnn/train/", "related_119861790_691021645147093_5752379036600259971_n.jpg"))
plot(image.test)

width <- 28
height <- 28

related_data <- extract_feature(dir_path = "/home/stathis/Desktop/itdrr2020/cnn/train/", width = width, height = height)
other_data <- extract_feature(dir_path = "/home/stathis/Desktop/itdrr2020/cnn/train/", width = width, height = height, is_related = FALSE)
dim(related_data)
saveRDS(related_data, "/home/stathis/Desktop/itdrr2020/cnn/related.5.june.2021.rds")
saveRDS(other_data, "/home/stathis/Desktop/itdrr2020/cnn/not.related.5.june.2021.rds")

complete.set <- rbind(related_data, other_data)
dim(complete.set)
training_index <- createDataPartition(complete.set$label, p = .7, times = 1)
training_index <- unlist(training_index)

train_set <- complete.set[training_index,]
dim(train_set)

test_set <- complete.set[-training_index,]
dim(test_set)

train_data <- data.matrix(train_set)
train_x <- t(train_data[, -1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))


## Model
mx_data <- mx.symbol.Variable('data')
## 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data = mx_data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2,2 ))
## 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
## 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
## 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
## Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

## Set seed for reproducibility
mx.set.seed(100)

## Set cpu device
device <- mx.cpu()

## Model training
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 30,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))


mx.model.save(model, prefix = "model_floods_related_cnn_6_june_2021", iteration = 30)

#classification on test array
predict_probs <- predict(model, test_array)
