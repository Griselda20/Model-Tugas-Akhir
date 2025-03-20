penggunaan memori sama,

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.00    Driver Version: 470.82.00    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:02:00.0 Off |                    0 |
| N/A   56C    P0   192W / 250W |  16965MiB / 22919MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P40           Off  | 00000000:81:00.0 Off |                    0 |
| N/A   39C    P0   182W / 250W |  16965MiB / 22919MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     21781      C   python                          16963MiB |
|    1   N/A  N/A     23580      C   python                          16963MiB |
+-----------------------------------------------------------------------------+


Waktu yang digunakan cenderung sama....

Train Progress:   2%|▎                      | 61/4000 [01:07<1:11:25,  1.09s/it, loss=7.92]

Train Progress:   2%|▏         | 60/4000 [01:07<1:12:57,  1.11s/it, loss=8.88]


dikarenakan biaya resource yang besar, tidak dijalankan lagi...