# Input parameters
# The number of asteroids processed in each Python program, e.g. 1000
batch_size=1000
# The number of batches run in parallel in each large job, e.g. 50
num_batch=50
# The largest asteroid number to process
max_ast_num=541130

# Compute number of jobs required to process all data
ast_per_job=$((num_batch*batch_size))
num_jobs=$((max_ast_num/ast_per_job + 1))
echo "Running $num_jobs jobs with batch_size=$batch_size and $num_batch batches per job..."

for ((job_num=1; job_num<=num_jobs; job_num++))
do
	bash scripts/asteroid_traj_batch.sh $batch_size $num_batch $max_ast_num $job_num
	# echo "job_num=$job_num"
done
