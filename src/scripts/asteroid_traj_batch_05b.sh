# Input parameters
batch_size=1000
num_batch=3
j0=450
sleep_time=0.1

# Set up index ranges
j1=$((j0+num_batch-1))
n0=$((j0*batch_size))
n1=$((j1*batch_size))
echo "Bash is processing asteroids from n0=$n0 to n1=$n1 with batch_size=$batch_size..."

# Run the first block of (batch_size-1) jobs in parallel
for (( i=0; i<num_batch-1; i++))
do
	j=$((j0+i))
	n=$((j*batch_size))
	if [$i<$((num_batch-1))]
	then
		python asteroids.py $n $batch_size &	
	fi
	sleep $sleep_time
	pids[i]=$!
	pid=$((pids[i]))
	echo "i=$i, n=$n, pid=$pid"
done

# Run the last job with a progress bar
sleep $sleep_time
python asteroids.py $n1 $batch_size --progress &
pids[num_batch-1]=$!
pid=$((pids[i]))
echo "i=$i, n=$n, pid=$pid"

# Wait for all outstanding jobs to be completed
for (( i=0; i<num_batch; i++))
do
	pid=$((pids[i]))
	wait $pid
	echo "process for i=$i, pid=$pid is complete."
done

echo "Done! Processed asteroid trajectories from $n0 to $n1."
