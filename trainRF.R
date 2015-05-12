# Train random forest classifier on Iris dataset in parallel settings

# clear memory
rm(list=ls());

# parallel computing
library(foreach);
library(doMC);
registerDoMC();
# random forest library
library('randomForest');

# Constants
NTREE = 20;
NODESIZE = 5;
MTRY = 2;
set.seed(42L); # because this is the answer!
# parallel execution
n_cores = parallel:::detectCores()/2;
ntree_per_core = floor(NTREE/n_cores);

# split dataset in train and test sets
TRAIN_RATIO = 0.8;
n = nrow(iris);
n_train = floor(TRAIN_RATIO*n);
n_test = n - n_train;
test_index = sample(1:n, n_test);
train_index = setdiff(1:n, test_index);

# train Random Forest in parallel
rf <- tryCatch({
     foreach(ntree=rep(ntree_per_core, n_cores), .combine=combine, .multicombine=TRUE, .packages='randomForest') %dopar% {
          randomForest(x=iris[train_index,1:4], y = iris$Species[train_index], ntree = ntree, nodesize = NODESIZE, mtry = MTRY);
     };     
}, error = function(err) {
     message(err);
});

# if training sucessful, predict the test set
if (!is.null(rf)) {
     # show features importance
     f_imp = varImpPlot(rf);
     
     # do prediction
     test_predicted = predict(rf, iris[test_index,1:4], type = "response");
     
     # validate results
     tt = table(iris$Species[test_index], test_predicted);
     accuracy = sum(diag(tt))/length(test_index);
     message('Training is complete. Acurracy is ', accuracy, ".");
};



