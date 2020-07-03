order="17 21 29 37 43 45 59 61 67 71 77 81 87 93 99 107 111 115 125"

for ((i=16; i<=128; i++))
do
	echo "Running img" $i
	./diff.sh $i
done
