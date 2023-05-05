#SCRIPT=`realpath $0`
#SCRIPTPATH=`dirname $SCRIPT`
#
#cd $SCRIPTPATH/..
#wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
#unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
#rm shapenetcore_partanno_segmentation_benchmark_v0.zip
#cd -






!pip install gdown
!gdown --id 1Vr7thc3GAXNTpDuMfNmB-I1THKkdmwFL --output shapenetcore_partanno_segmentation_benchmark_v0.zip
!unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
!rm shapenetcore_partanno_segmentation_benchmark_v0.zip
