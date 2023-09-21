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


############################ GSE132364 ############################
zcat GSE132364_rawData_subset.txt.gz | wc -l  # 19697 (cell_num=19697-1=19696)

## not have correspondence ##
# E12.5 live
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP18.live | wc -l  # 71
# E13.5 live
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP12.live | wc -l   # 174
# E13.5 GFP
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP12.GFP | wc -l   # 252
# E14.5 live
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP14.live | wc -l   # 114
# E14.5 GFP
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP14.GFP | wc -l   # 209
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP16.GFP | wc -l   # 275
# E15.5 live
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP11.live | wc -l  # 51
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP10.live | wc -l  # 51
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP17.live | wc -l  # 67
# E15.5 GFP
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP11.GFP | wc -l   # 278
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP10.GFP | wc -l   # 325
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP18.GFP | wc -l   # 61
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP17.GFP | wc -l   # 125
# E18.5 live
zcat GSE132364_rawData_subset.txt.gz | head -n 1 | xargs -n1 | grep EP16.live | wc -l  # 44
###############################

load('GSE132364_seurat_full.Rdata')
ls()
str(full.seurat)















