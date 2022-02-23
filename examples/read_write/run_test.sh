for i in `seq 1 20`; do
    DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py data/1G/$i  --test_all
done  
