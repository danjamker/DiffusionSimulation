#!/usr/bin/env bash
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/reddit/community-split/train ./data/reddit-community-split-train
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/reddit/community-split/test ./data/reddit-community-split-test
cat ./data/reddit-community-split-train ./data/reddit-community-split-test > ./data/reddit-community-split
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkX.py -r hadoop -v ./data/reddit-community-split --network ./networks/reddit_traversal_network.gpickle -o ./output/reddit-community-innovation --no-output --conf-path .mrjob.conf
