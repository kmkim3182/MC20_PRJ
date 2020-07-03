for ((i=16; i<=128; i++))
do
	echo "Running img" $i
	./diff.sh $i
done
