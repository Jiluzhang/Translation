############################ GSE132184 ############################
tar -xvf GSE132184_RAW.tar 

for GSM in GSM3852665 GSM3852666 GSM3852667 GSM3852668;do
	tar -xzvf `echo $GSM\*filtered_feature_bc_matrix.tar.gz`
	mkdir -p /fs/DC/mouse/raw_mat_meta/GSE132/GSE132184/$GSM
	mv barcodes.tsv.gz features.tsv.gz matrix.mtx.gz /fs/DC/mouse/raw_mat_meta/GSE132/GSE132184/$GSM
	rm `echo $GSM\*filtered_feature_bc_matrix.tar.gz`
	echo $GSM done
done

## cell and gene number stats
for GSM in GSM3852665 GSM3852666 GSM3852667 GSM3852668;do
	echo $GSM cell_num:`zcat $GSM/barcodes.tsv.gz | wc -l` gene_num:`zcat $GSM/features.tsv.gz | wc -l`
done


############################ GSE132272 ############################
tar -xvf GSE132272_RAW.tar

for GSM in GSM3855433 GSM3855434;do
 	mv `echo $GSM\*barcodes*` barcodes.tsv.gz
  	zcat `echo $GSM\*genes*` | awk '{print $0 "\t" "Gene Expression"}' > features.tsv && gzip features.tsv && rm `echo $GSM\*genes*`  # add "Gene Expression" to the column
   	mv `echo $GSM\*matrix*` matrix.mtx.gz
	mkdir -p /fs/DC/mouse/raw_mat_meta/GSE132/GSE132272/$GSM
	mv barcodes.tsv.gz features.tsv.gz matrix.mtx.gz /fs/DC/mouse/raw_mat_meta/GSE132/GSE132272/$GSM
	echo $GSM done
done

## cell and gene number stats
for GSM in `ls`;do
	echo $GSM cell_num:`zcat $GSM/barcodes.tsv.gz | wc -l` gene_num:`zcat $GSM/features.tsv.gz | wc -l`
done





