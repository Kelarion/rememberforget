#!/bin/bash
# 
# Send all my jobs in a swarm

L=3
for P in 10 100 1000
do
	for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0 3.0
	do
		jobfile="runjob.sh"
		echo "submitting with P $P and N $p"

		echo "#!/bin/sh" > $jobfile
		echo "#SBATCH --account=theory" >> $jobfile
		echo "#SBATCH --job-name=geom" >> $jobfile
		echo "#SBATCH -c 1" >> $jobfile
		echo "#SBATCH --time=29:00" >> $jobfile
		echo "#SBATCH --mem-per-cpu=1gb" >> $jobfile

		echo "module load anaconda/3-5.1" >> $jobfile
		echo "source activate pete" >> $jobfile
		
		echo "python remember-forget/habaexperiment.py -p $P -n $p -l $L" >> $jobfile

		echo "date" >> $jobfile

		sbatch $jobfile
		echo "waiting"
		sleep 1s
	done
done

