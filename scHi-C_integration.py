## SEE github: https://github.com/4dglab/SEE

# conda env create -f environment.yml
# conda activate see


for file in *pairs.gz; do
    s="${file%.pairs.gz}"
    cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 mm10.chrom.sizes:10000 $file $s.cool
    echo "$s" done
done
