# Function with timestamp
timestamp() 
{
  date +"%Y-%m-%d %H:%M:%S"
}

# timestamp
echo "Hello, the time is $(date +"%Y-%m-%d %H:%M:%S")"
