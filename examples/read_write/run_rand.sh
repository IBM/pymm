for i in `seq 1 20`; do

     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=100000 --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_10GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=10000 --write_size=2097152
done

for i in `seq 1 20`; do
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=100000 --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_1GB/$i_pymm_mempnvme_$i --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=10000 --write_size=2097152
done

for i in `seq 1 5`; do
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=1000000 --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=1000000 --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=1000000 --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=1000000 --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=1000000 --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=100000 --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --num_write=10000 --write_size=2097152
done

for i in `seq 1 10`; do
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=100000 --persist_at_the_end --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/write_rand_persist_once_10GB/$i_pymm_mempnvme_$i --test_all --num_write=10000 --persist_at_the_end --write_size=2097152
done

for i in `seq 1 10`; do
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=1000000 --persist_at_the_end --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=100000 --persist_at_the_end --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/write_rand_persist_once_1GB/$i_pymm_mempnvme_$i --test_all --num_write=10000 --persist_at_the_end --write_size=2097152
done

for i in `seq 1 5`; do
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i   --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=1000000 --persist_at_the_end --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=1000000 --persist_at_the_end --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=1000000 --persist_at_the_end --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=1000000 --persist_at_the_end --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=1000000 --persist_at_the_end --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=100000 --persist_at_the_end --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 100 data/write_rand_persist_once_100GB/$i_pymm_mempnvme_$i --pymm_dev_dax0  --pymm_fs_dax0  --numpymemmap_nvme --numpymemmap_fs_dax0 --numpy_save_nvme  --numpy_save_fs_dax0 --dram --num_write=10000 --persist_at_the_end --write_size=2097152
done


