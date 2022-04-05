for i in `seq 1 20`; do

     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=100000 --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 10 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=10000 --write_size=2097152
done


for i in `seq 1 20`; do

     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=8
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=64
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=512
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=4096
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=1000000 --write_size=32768
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=100000 --write_size=262144
     sudo DAX_RESET=1 numactl --cpunodebind=0 python3 write_rand_different_media_do_work.py 1 data/a --pymm_dev_dax0 --pymm_dev_dax1 --pymm_fs_dax0 --pymm_fs_dax1 --pymm_nvme --numpymemmap_nvme --num_write=10000 --write_size=2097152
done

