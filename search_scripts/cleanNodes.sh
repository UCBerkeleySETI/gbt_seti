for (( i=1; i<10; i++))
do
    ssh node00$i rm -fr /local/amag0001 
    echo "node00$i done"
done

for (( i=10; i<61; i++))
do
    ssh node0$i rm -fr /local/amag0001 
    echo "node0$i done"
done
