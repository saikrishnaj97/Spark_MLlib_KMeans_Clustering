import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.File;
import java.util.List;


public class K_Means_Clustering {

    public List<Tuple2<String, Integer>> build_K_means_clustering(JavaSparkContext sparkContext){

        // relative path
        File txt_file = new File("twitter2D.txt");

        // absolute path to our data file
        String abs_path = txt_file.getAbsolutePath();
        // reading the text file
        JavaRDD<String> data = sparkContext.textFile(abs_path);

        //RDD of type Tuple2<Vector, String>
        JavaRDD<Tuple2<Vector, String>> parsed_data = data.map(line -> {
                    String[] array = line.split(","); //Iterating through each line and splitting by ","

                    //getting the coordinates of each tweet(line) into a array of type double
                    double[] values = new double[2];
                    for (int i = 0; i < 2; i++)
                        values[i] = Double.parseDouble(array[i]);

                    //returning a tuple where vector contains world coordinates and string has tweet
                    return new Tuple2<Vector, String>(Vectors.dense(values), array[array.length-1]);
                }
        );

        //caching parsed_data for reuse it for training the algorithm
        parsed_data.cache();

        // Cluster the data into four classes using KMeans
        int numClusters = 4;
        int numIterations = 20;

        // Training the KMeans algorithm with the world coordinates vector
        KMeansModel model = KMeans.train(parsed_data.map(p -> p._1).rdd(), numClusters, numIterations);

        /*
         Predicting the cluster index to each tweet and returning a list of tuple with the
         string containing tweet and integer containing the corresponding cluster index
        */
        List<Tuple2<String, Integer>> predictions = parsed_data.map(line ->{
            int cluster = model.predict(line._1);
            return new Tuple2<String, Integer>(line._2, cluster);
        }).sortBy(a -> a._2, true, 1).collect(); 	// sorting the list by cluster index

        return predictions;

    }
}
