# Document classification with Apache Spark on an American Classic

These are files tokenized and put into passages of length=10: 

./mock.tokens

./watch.tokens


### These are the raw first chapters from which the modeling files are created: 
./mock1.txt

./watch1.txt

### These are the locations of the scala files you will need to modify:
./src/main/scala/Stemmer.scala - utility to tokenize the data, comes from here:

https://chimpler.wordpress.com/2014/06/11/classifiying-documents-using-naive-bayes-on-apache-spark-mllib/

./src/main/scala/Models.scala - reads data, build models, report results

### Step 1: Compiling the code - create a new jar:

1. Start by downloading sbt (http://www.scala-sbt.org/download.html)
2. Make sure these files are in place ./model.sbt and ./project/plugins.sbt
3. Once sbt is installed and the files are ready, issue this command: >sbt assembly
4. If successful, the following jar should be created: target/scala-2.10/modeling-assembly-0.0.1.jar
5. Do this any time you make changes to the code (this code won't compile on your machine unless you change the file locations)

### Step 2: Building and evaluating the models:
1. Get Spark - I used version 1.3.1 (http://spark.apache.org/downloads.html)
2. Make sure the locations in the file have been modified and you're pointing to your files.
3. If you're trying to run the ROC's, make sure you delete the ROC folder first or you will get an error
4. Once the jar is created, this command will run the models and produce the results like those in the blog:

>~/spark-1.3.1-bin-hadoop2.4/bin/spark-submit --class "Models" target/scala-2.10/modeling-assembly-0.0.1.jar  > output

Note: depending on your spark configs, you make see a lot of logs from the Spark program

### Sample output (results differ slightly due to randomness):
(Number of training examples: ,419)

(Number of test examples: ,167)

Training Naive Bayes Model...

(NB Training accuracy: ,0.9212410501193318)

(NB Test set accuracy: ,0.7844311377245509)

Naive Bayes Confusion Matrix:

(Predict:mock,label:mock -> ,89)

(Predict:watch,label:watch -> ,42)

(Predict:mock,label:watch -> ,23)

(Predict:watch,label:mock -> ,13)

Training Random Forest Regression Model...

(RF Training AuROC: ,0.980446863231377)

(RF Test AuROC: ,0.8358974358974359)

Attempting to write RF ROC to file...

Training Gradient Boosted Trees Regression Model...

(GB Training AuROC: ,0.9989207311472399)

(GB Test AuROC: ,0.8264705882352942)

Attempting to write GB ROC to file...

### Ideas for Experimentation:
1. Experiment with more trees, max depth for the model building - you could also modify code to take an argument
2. Use the spark shell to experiment with the tokenizer and understand how it works.
3. Look at the MLLib documentation for the other model metrics - which make sense?
4. Grab another source of documents like your favorite book. Prepare the data and determine how that stacks up.

