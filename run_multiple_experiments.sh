#!/bin/bash
R=$1
SP=$2
set --

/usr/bin/sbatch cluster_runner.sh 11 R"$R"_all_mjj-translation_10000 "$SP"
/usr/bin/sbatch cluster_runner.sh 11 R"$R"_all_mjj-translation_1000 "$SP"
/usr/bin/sbatch cluster_runner.sh 11 R"$R"_all_mjj-translation_5000 "$SP"
/usr/bin/sbatch cluster_runner.sh 11 R"$R"_all_mjj-translation_100 "$SP"
#/usr/bin/sbatch cluster_runner.sh 11 R"$R"_all "$SP"
#/usr/bin/sbatch cluster_runner.sh 4 R"$R"_anode "$SP"
#/usr/bin/sbatch cluster_runner.sh 4 R"$R"_salad "$SP"
#/usr/bin/sbatch cluster_runner.sh 1 R"$R"_mjj "$SP"
#/usr/bin/sbatch cluster_runner.sh 2 R"$R"_mjj_m1 "$SP"
#/usr/bin/sbatch cluster_runner.sh 2 R"$R"_mjj_m1minusm2 "$SP"
#/usr/bin/sbatch cluster_runner.sh 3 R"$R"_mjj_m1_m1minusm2 "$SP"
