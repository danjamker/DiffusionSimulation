#!/usr/bin/env bash
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/twitter/geo-split/train ./data/twitter-geo-split-train
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/twitter/geo-split/test ./data/twitter-geo-split-test
cat ./data/twitter-geo-split-train ./data/twitter-geo-split-test > ./data/twitter-geo-split
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkX.py -r hadoop -v ./data/twitter-geo --network ./networks/twitter_geo_network.gpickle -o ./output/twitter-geo-innovation --no-output --conf-path .mrjob.conf
