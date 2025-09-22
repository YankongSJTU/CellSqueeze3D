python Merge_JSON.py Demodata/
cut -d "-" -f 1-3   Demodata/sample_entropy_analysis.csv >tmpdata1
cut -d "," -f 2-   Demodata/sample_entropy_analysis.csv >tmpdata2
paste tmpdata1 tmpdata2 |sed 's/\t/,/g' >result.csv
python  cal_entropys.py 
