for ((i=16; i<=128; i++))
do
 N=$i CASE="run$i" make validation
done
