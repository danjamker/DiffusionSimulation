#!/usr/bin/env bash
python GenerateFileList.py http://scc-culture-mind.lancs.ac.uk:50070/user/kershad1/output/reddit/innovation/2013-5 ./Data/2013.txt
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkX.py -r hadoop -v ./Data/2013.txt --network ./Data/2013.gpickle -o ./innovation-avr/2013 --no-output --conf-path .mrjob.conf
