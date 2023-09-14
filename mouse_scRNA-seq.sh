## GSE132184
tar -xvf GSE132184_RAW.tar 

for GSM in GSM3852665 GSM3852666 GSM3852667 GSM3852668;do
	tar -xzvf `echo $GSM\*filtered_feature_bc_matrix.tar.gz`
	mkdir -p /fs/DC/mouse/raw_mat_meta/GSE132/GSE132184/$GSM
	mv barcodes.tsv.gz features.tsv.gz matrix.mtx.gz /fs/DC/mouse/raw_mat_meta/GSE132/GSE132184/$GSM
	rm `echo $GSM\*filtered_feature_bc_matrix.tar.gz`
	echo $GSM done
done

meta data???
