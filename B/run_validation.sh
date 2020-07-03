slots="4"
nodes="1"
order="17 21 29 37 43 45 59 61 67 71 77 81 87 93 99 107 111 115 125"

cd ../common/

#for ((i=16; i<=128; i++))
#do
#	./genrgb $i
#done

cd -
#: << "END"
for ((i=16; i<=128; i++))
do
 SLOTS=$slots NODES=$nodes N=$i make run
done
#END
