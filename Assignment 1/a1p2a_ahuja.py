from random import random
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    data = [(1, "The horse raced past the barn fell"),
            (2, "The complex houses married and single soldiers and their families"),
            (3, "There is nothing either good or bad, but thinking makes it so"),
            (4, "I burn, I pine, I perish"),
            (5, "Come what come may, time and the hour runs through the roughest day"),
            (6, "Be a yardstick of quality."),
            (7, "A horse is the projection of peoples' dreams about themselves - strong, powerful, beautiful"),
            (8, "I believe that at the end of the century the use of words and general educated opinion will have altered so much that one will be able to speak of machines thinking without expecting to be contradicted."),
		    (9, "The car raced past the finish line just in time."),
		    (10, "Car engines purred and the tires burned.")]
    rdd = sc.parallelize(data)

    # Word Count implementation below
    wordCounts = rdd.map(lambda k: k[1]).flatMap(lambda k: k.split(' ')).map(lambda k: (k.lower(),1)).reduceByKey(lambda a, b: a + b).collect()
    print(wordCounts)




    # Set difference implementation below
    print("\n\n*****************\n Set Difference\n*****************\n")
    data = [('R', [x for x in range(50) if random() > 0.5]), ('S', [x for x in range(50) if random() > 0.75])]
    #data = [('R', [x for x in range(10) if random() > 0.5]), ('S', [x for x in range(10) if random() > 0.75])]
    rdd = sc.parallelize(data)
    #setDiff = rdd.flatMap(lambda k: [(v, k[0]) for v in k[1]]).groupByKey().map(lambda k: ('R' in k[1] and 'S' not in k[1], k[0])).filter(lambda k: k[0]).map(lambda k: k[1]).collect()
    #setDiff = [('R',setDiff)]
    #setDiff = rdd.flatMap(lambda k: [(v, k[0]) for v in k[1]]).groupByKey().map(lambda k: ('R' in k[1] and 'S' not in k[1], k[0])).groupByKey().filter(lambda k: k[0]).map(lambda k:('R',list(k[1]))).collect()
    setDiff = rdd.flatMap(lambda k: [(v, k[0]) for v in k[1]]).groupByKey().map(lambda k: ('R' in k[1] and 'S' not in k[1], k[0])).groupByKey().filter(lambda k: k[0]).map(lambda k:tuple(k[1])).collect()

    print(data)
    print(setDiff)
