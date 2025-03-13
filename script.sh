training_dir=$1

for i in "${training_dir}"/*
do
    touch "${i}"/"test"
done