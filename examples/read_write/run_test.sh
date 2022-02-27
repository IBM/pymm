#for i in `seq 1 20`; do
#    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 1 data/write_seq_1GB/$i --test_all
#    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 10 data/write_seq_10GB/$i --test_all
#done 

 for i in `seq 1 20`; do
    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 150 data/write_seq_150GB/$i --dram --pymm_fs_dax0  --pymm_dev_dax0  --pymm_nvme --numpy_save_nvme
    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 150 data/write_seq_150GB/numpy_save_fs_dax$i --numpy_save_fs_dax0
    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 100 data/write_seq_100GB/numpy_save_fs_dax$i --numpy_save_fs_dax0
done  
