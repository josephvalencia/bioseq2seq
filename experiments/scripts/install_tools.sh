# CPAT
pip install CPAT

# CPC2
wget http://cpc2.gao-lab.org/data/CPC2-beta.tar.gz
export DIR="$PWD"
gzip -dc CPC2-beta.tar.gz | tar xf -
cd CPC2-beta
export CPC_HOME="$PWD"
cd libs/libsvm
gzip -dc libsvm-3.18.tar.gz | tar xf -
cd libsvm-3.18
make clean && make
cd $DIR

# rnasamba
conda create -n rnasamba_env
conda activate rnasamba_env
conda install -c conda-forge -c bioconda rnasamba

