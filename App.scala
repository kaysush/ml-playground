import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, StandardScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object App {

  def parseValues(value : String) : Double = {

    value match {
      case "Male" | "France" => 0.0
      case "Female" | "Spain" => 1.0
      case "Germany" => 2.0
      case default => value.toDouble

    }

  }

  def parseLine(line : String) = {

    var fields = line.split(",")
    var vector = fields.slice(4,13).map(parseValues)
    LabeledPoint(parseValues(fields(13)), Vectors.dense(vector))

  }

  def main(args : Array[String]): Unit = {

    val session = SparkSession.builder().appName("ANNDemo").master("local[*]").getOrCreate()
    val lines = session.sparkContext.textFile("C:\\ScalaSpark\\Churn_Modelling.csv")
    import session.implicits._
    val rdd = lines.map(parseLine).toDS()
    val scaler = new StandardScaler()
                                .setInputCol("features")
                                .setOutputCol("scaledFeatures")
                                  .setWithMean(true)
                                  .setWithStd(true)
    val scalerModel = scaler.fit(rdd)

    val data = scalerModel.transform(rdd)



    val splits = data.randomSplit(Array(0.75, 0.25), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val layers = Array[Int](9, 5, 5, 5, 5, 5, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

  }

}
