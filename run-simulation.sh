#!/usr/bin/env bash
python GenerateSequenceFile.py 1 10 ./data/loops.txt
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkXSimulations.py -r hadoop -v ./data/loops.txt --network ./networks/twitter_mention_network.gpickle -o ./output/twitter_mention_network_simulation --no-output --conf-path ./etc/mrjob.conf
python MRJobNetworkXSimulations.py -r hadoop -v ./data/loops.txt --network ./networks/twitter_geo_network.gpickle -o ./output/twitter_geo_network_simulation --no-output --conf-path ./etc/mrjob.conf
python MRJobNetworkXSimulations.py -r hadoop -v ./data/loops.txt --network ./networks/reddit_traversal_network.gpickle -o ./output/reddit_traversal_network_simulation --no-output --conf-path ./etc/mrjob.conf
python MRJobNetworkXSimulations.py -r hadoop -v ./data/loops.txt --network ./networks/reddit_comment_network.gpickle -o ./output/reddit_comment_network_simulation --no-output --conf-path ./etc/mrjob.conf
