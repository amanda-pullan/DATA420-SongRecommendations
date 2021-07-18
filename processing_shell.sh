# DATA420-21S1 Assignment 2
# Data Processing Q1

# exploring the datasets

# msd
hdfs dfs -ls /data/msd

# code to generate directory tree
hdfs dfs -ls -R /data/msd | awk '{print $8}' | sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/' | hdfs dfs -appendToFile - hdfs:///user/apu20/a2_tree.txt
hdfs dfs -copyToLocal /user/apu20/ ~/a2_tree.txt

# -----------------------------------------------------------------------------------------------------------------

# audio
hdfs dfs -ls /data/msd/audio

# attributes contains files saved as .csv
hdfs dfs -cat /data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv | head
hdfs dfs -cat /data/msd/audio/attributes/msd-jmir-lpc-all-v1.0.attributes.csv | head

# features contains sub directories (called .csv) containing .csv.gz part files
# first line is a directory
# part file contains numerical values
# contains track ID
hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00000.csv.gz | gunzip | head

# statistics
# contains track ID, title, artist etc
# first row is header
hdfs dfs -cat /data/msd/audio/statistics/sample_properties.csv.gz | gunzip | head

# -----------------------------------------------------------------------------------------------------------------

# genre
# files contains track ID, genre
hdfs dfs -cat /data/msd/genre/msd-MASD-styleAssignment.tsv | head
hdfs dfs -cat /data/msd/genre/msd-topMAGD-genreAssignment.tsv | head # smaller set of main genres
hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv | head # larger set of genres

# -----------------------------------------------------------------------------------------------------------------

# main/summary
# both have header row
hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz | gunzip | head # contains track ID and more
hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | head # contains song ID and more

# -----------------------------------------------------------------------------------------------------------------

# tasteprofile

# mismatches
hdfs dfs -cat /data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt | head
hdfs dfs -cat /data/msd/tasteprofile/mismatches/sid_mismatches.txt | head

# triplets.tsv
# format: userID, trackID, count
hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/part-00000.tsv.gz | gunzip | head













