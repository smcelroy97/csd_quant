#!/bin/bash
#$ -cwd
#$ -N csd_analysis
#$ -q cpu.q
#$ -pe smp 40
#$ -l h_vmem=200G
#$ -l h_rt=2:40:00
#$ -o erp_run.out
#$ -e erp_run.err

source ~/.bashrc
conda activate csd_quant
export UCX_TLS=tcp,self
mpiexec -n "$NSLOTS" python csd_pca.py