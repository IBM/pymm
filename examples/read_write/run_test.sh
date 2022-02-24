#for i in `seq 1 20`; do
for i in `seq 1 4`; do
#    DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py data/100GB/$i  --test_all
    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 50 data/write_seq_50GB_wbinvd/$i --numa_local
   
done  
