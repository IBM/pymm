#sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 300 data/a$i --test_all --no_remove
#exit(0)
for i in `seq 1 10`; do
#    sudo numactl --cpunodebind=0 --membind=0 python3 filter_different_media_do_work.py 50 data/filter_50GB/filter$i --test_all
     sudo numactl --cpunodebind=0 --membind=0 python3 filter_different_media_do_work.py 300 data/filter_300GB/pymm_nvme$i --pymm_nvme
done 

# for i in `seq 1 1`; do
#    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 150 data/write_seq_150GB/$i --dram --pymm_fs_dax0  --pymm_dev_dax0  --pymm_nvme --numpy_save_nvme
#    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 150 data/write_seq_150GB/numpy_save_fs_dax$i --numpy_save_fs_dax0
#    sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py 100 data/write_seq_100GB/numpy_save_fs_dax$i --numpy_save_fs_dax0
#done  
