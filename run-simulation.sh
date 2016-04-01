#!/usr/bin/env bash
python GenerateSequenceFile.py 1 100 ./Data/loops.txt
tar -zcvf compressed-code.tar.gz .
python MRJobNetworkXSimulations.py -r hadoop -v ./Data/loops.txt --network ./Network/Twitter-Geo-Norm-Self-Loop-protocol-2.gpickle -o ./output/twitter-geo --no-output --conf-path .mrjob.conf
