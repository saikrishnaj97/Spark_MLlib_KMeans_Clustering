import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.List;

public class caller {
    public static void main(String[] args)
    {

        System.setProperty("hadoop.home.dir", "C:/winutils");
        SparkConf sparkConf = new SparkConf()
                .setAppName("LSDA_Assignment4") //setting appname to uniquely recognise job in a cluster(isn't much relevant to our task here)
                .setMaster("local[4]").set("spark.executor.memory", "1g"); //4 core processor to work individually with 1 gigabyte of heap memory

        //creating JavaSparkContext object to start the spark session
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        K_Means_Clustering s1=new K_Means_Clustering();
        List<Tuple2<String, Integer>> predictions= s1.build_K_means_clustering(sparkContext);

        // Printing the results
        System.out.println("\n");
        for (Tuple2<String, Integer> element : predictions) {
            System.out.println("Tweet \"" + element._1 + "\" is in cluster " + element._2);
        }
        System.out.println("\n");

        //We can also print the results as below
        //predictions.stream().map(element -> "Tweet \"" + element._1 + "\" is in cluster " + element._2).forEach(System.out::println);


        //ending the spark session
        sparkContext.stop();
        sparkContext.close();
    }
}
