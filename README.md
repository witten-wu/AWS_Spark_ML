## AWS_Spark_ML

### Environment
AWS EMR cluster with Spark 2.4.0

### Experiment
In this Experiment, I write a machine learning program under the Spark framework, by using the Python language with pyspark library that can classify samples from the MNIST dataset using library pyspark, where the MNIST dataset is a handwriting digital dataset. It contains 60,000 training samples and 10,000 testing samples. Each sample is a 28*28 black- white image of a digital. Such as these:
<img width="626" alt="image" src="https://user-images.githubusercontent.com/38986230/171347455-52cc32e4-6f10-408f-969b-8e7ad5c0d9d8.png">

### DataSet
MNIST dataset Can download the original dataset from the official website http://yann.lecun.com/exdb/mnist/

<img width="539" alt="image" src="https://user-images.githubusercontent.com/38986230/171347533-cd46e7ce-edcd-4310-ab8e-2e292e430a10.png">
In the .csv file, each row represents one sample, which consists of 785 columns, where the 1st column is the label (0~9), and the rest of 784 (=28*28) columns correspond to 784 pixels. In other words, each sample in the MNIST dataset can be represented by a 784-dimensional vector.
