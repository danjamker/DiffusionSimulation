#!/usr/bin/env bash
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/twitter/user-split/train ./data/twitter-user-split-train
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/twitter/user-split/test ./data/twitter-user-split-test
cat ./data/twitter-user-split-train ./data/twitter-user-split-test > ./data/twitter-user-split
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkX.py -r hadoop -v ./data/twitter-user-split --network ./networks/twitter_mention_network.gpickle -o ./output/twitter-mention-innovation --no-output --conf-path ./etc/mrjob.conf
