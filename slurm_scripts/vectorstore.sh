#!/bin/bash

#####################################################
##        Important parameters of your job         ##
##             are specified here 		           ##
#####################################################

#SBATCH --time=4:00:00				        ## total computing time
#SBATCH --nodes=1				            ## number of nodes 
#SBATCH --ntasks-per-node=1			        ## number of tasks per node
#SBATCH --cpus-per-task=16			        ## number of CPUs per task
#SBATCH --mem=256GB				            ## memory per node
#SBATCH --partition=secondary			    ## queue
#SBATCH --output=vector_store.out		    ## file that will receive output from execution
#SBATCH --error=vector_store.err		    ## file that will receive any error messages
#SBATCH --mail-user=vikramr2@illinois.edu
#SBATCH --mail-type=BEGIN,END

########## Run your executable ######################

PDF_DIR=/projects/illinois/eng/cs/chackoge/illinoiscomputes/vikramr2/llm-crispr/data/pdfs

python3 ../rag/vectordb.py --pdf_dir ${PDF_DIR} --output_dir ../rag/vectorstore 
