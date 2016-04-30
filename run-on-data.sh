#!/usr/bin/env bash
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/reddit/user-split/train ./date/reddit-user-split-train
python ./tools/GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/reddit/user-split/test ./date/reddit-user-split-test
cat ./data/reddit-user-split-train ./data/reddit-user-split-test > ./data/reddit-user-split2
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkX.py -r hadoop -v ./data/reddit-user-split --network ./data/reddit_traversal_network.gpickle -o ./output/reddit-traversal-innovation --no-output --conf-path .mrjob.conf
