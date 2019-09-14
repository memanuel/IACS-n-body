# Extract command line arguments
# This is the number of asteroids processed in each Python program, e.g. 1000
batch_size=$1
# The number of batches run in parallel in each large job, e.g. 50
num_batch=$2
# The largest asteroid number to process
max_ast_num=$3
# The number of this job, starting from 1
job_num=$4

# Additional parameters
sleep_time=0.1

# Set up index ranges
# j is the multiplier of the batch size; ranges from j0 to j1
j0=$((num_batch*(job_num-1)))
j1=$((j0+num_batch-1))
# n is the first asteroid number to process in each call; ranges from n0 to n1
n0=$((j0*batch_size))
n1=$((j1*batch_size))
echo "Bash asteroid_traj_batch.sh is processing asteroids from n0=$n0 to n1=$n1 with batch_size=$batch_size..."

# Run all the jobs jobs in parallel
for (( i=0; i<num_batch; i++))
do
	# j is the multiplier of the batch size
	j=$((j0+i))
	# n is is the asteroid number for the current python job
	n=$((j*batch_size))
	# if n is past the largest known asteroid, terminate the loop early
	if [ $n -gt $max_ast_num ]
	then
		break
	fi

	# run the first job with a progress bar, the rest silently
	if [ $i -lt $((num_batch-1)) ]
	then		
		python asteroids.py $n $batch_size &	
	else
		python asteroids.py $n $batch_size --progress &
	fi

	# Slight pause so the batches will be executed in the specified order
	sleep $sleep_time
	# Save the process ID
	pids[i]=$!
	pid=$((pids[i]))
	# echo "i=$i, n=$n, pid=$pid"
done

# Wait for all outstanding jobs to be completed
for (( i=0; i<num_batch; i++))
do
	j=$((j0+i))
	n=$((j*batch_size))
	pid=$((pids[i]))
	wait $pid
	echo "Process for i=$i, n=$n, pid=$pid is complete."
done

echo -e "\n********************************************************************************"
echo "$(date +"%Y-%m-%d %H:%M:%S") Done! Processed asteroid trajectories from $n0 to $n1."
echo -e "********************************************************************************\n"
