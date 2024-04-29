# Importing necessary dependencies
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce


# Creating a Spark Session 
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Checking for commandline arguments passed
# If a file is passed via commandline, use that for prediction using model
# Else throw an error

try:
	filepn = "/job/"+ str(sys.argv[1])
	data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
	print("***********************************************************************")
	print ("File input :", str(sys.argv[1]))
	print("***********************************************************************")
	
except:
	print("***********************************************************************")
	print("Test File cannot be found/File not passed. Please try again with following parameters\n")
	print("Pass parameters as: ")
	print("-----------------------------------------------------------------")
	print("$sudo docker run -v <host-folder-path>:/job kruthika547nayak/winetest:latest <Name of csv file> ")
	print("-----------------------------------------------------------------")
	print("--> Make sure that the <host-folder-path> contains the CSV as well as Model file for prediction")
	print("***********************************************************************")
	exit()

#To clean out CSV headers if quotes are present
old_column_name = data_test.schema.names
print(data_test.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_test)
print(data_test.schema)




# Create a PipelineModel object to load saved model parameters from Train

try:
	PipeModel = PipelineModel.load("/job/Modelfile")
except:
	print("***********************************************************************")
	print("Model file cannot be found. Please check whether model file is present in the directory of mount\n")
	print("Pass parameters as: ")
	print("-----------------------------------------------------------------")
	print("$sudo docker run -v <host-folder-path>:/job kruthika547nayak/winetest:latest TestDataset.csv ")
	print("-----------------------------------------------------------------")
	print("--> Make sure that the <host-folder-path> contains the CSV as well as Model file for prediction")
	print("***********************************************************************")
	exit()


# Generate predictions for Input dataset file
try:
	test_prediction = PipeModel.transform(data_test)
except:
	print("***********************************************************************")
	print ("Please check CSV file : labels may be improper")
	print("***********************************************************************")
	

#test_prediction.printSchema()

# Save the resulting predictions with original datset to a CSV File
test_prediction.drop("feature", "Scaled_feature", "rawPrediction", "probability").write.mode("overwrite").option("header", "true").csv("/job/resultdata.csv")
#test_prediction.select("quality", "prediction").write.mode("overwrite").option("header", "true").csv("/job/resultdata")

# Creating a evaluator classification object to generate metrics for predictions
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol = "prediction")

# Calculating the F1 score/Accuracy of the model with Test dataset
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

print("***********************************************************************")
print("++++++++++++++++++++++++++++++ Metrics ++++++++++++++++++++++++++++++++")
print("***********************************************************************")
print("[Test] F1 score = ", test_F1score)
print("[Test] F1 score = ", test_accuracy)
print("***********************************************************************")

# Save the results onto a Text File called results.txt
fp = open("/job/results.txt", "w")
fp.write("***********************************************************************\n")
fp.write("[Test] F1 score =  %s\n" %test_F1score)
fp.write("[Test] F1 score =  %s\n" %test_accuracy)
fp.write("***********************************************************************\n")

# Closing the file
fp.close()
