# Extract the command line argument: whick chunk of batches to run
chunk_num=$1

# Input parameters
batch_size=1000
num_batch=50
sleep_time=0.1

# Compute j0 from the command line argument
# echo "chunk_num = $chunk_num"
# j0=$((chunk_num*num_batch))
# echo "j0=$j0"
# exit

# Set up index ranges
# j is the multiplier of the batch size
j0=$((chunk_num*num_batch))
j1=$((j0+num_batch-1))
# n is the first asteroid number to process in each call; 
# n0=$((j0*batch_size))
# n1=$((j1*batch_size))
echo "Bash is processing asteroids from n0=$n0 to n1=$n1 with batch_size=$batch_size..."

# Run all the jobs jobs in parallel
for (( i=0; i<num_batch; i++))
do
	# j is the multiplier of the batch size
	j=$((j0+i))
	# n is is the asteroid number for the current python job
	n=$((j*batch_size))
	# run all the jobs except the last one without a progress bar
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

echo "Done! Processed asteroid trajectories from $n0 to $n1."
