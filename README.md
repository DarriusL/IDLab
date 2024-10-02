# IDLab

Code for Identity recognition and intrusion detection in wireless sensing.

**Dataset acquisition:**
Wait for the public release

## Command

### usage

```shell
usage: executor.py [-h] [--config CONFIG] [--saved_config SAVED_CONFIG] [--auto_shutdown AUTO_SHUTDOWN]
```

### options

```shell
-h, --help            show this help message and exit
--config CONFIG, -cfg CONFIG
                    config for run
--saved_config SAVED_CONFIG, -sc SAVED_CONFIG
                    path for saved config to test
--auto_shutdown AUTO_SHUTDOWN, -as AUTO_SHUTDOWN
                    automatic shutdown after program completion
```

### quick start
```shell
python executor.py -cfg=path/to/config
```

## Model

- [x] Gait-Enhance[[1](#ref1)]
- [x] Deep-WiID[[2](#ref2)]
- [x] Caution[[3](#ref3)]
- [x] CSIID[[4](#ref4)]
- [x] Gate-ID[[6](#ref5)]
- [x] WiAU[[7](#ref6)]
- [x] WiAi-ID[[8](#ref7)]
- [x] Bird



# Refrence

1. <a name="ref1"></a>Yang J, Liu Y, Wu Y, et al. Gait-Enhance: Robust Gait Recognition of Complex Walking Patterns Based on WiFi CSI[C]//2023 IEEE Smart World Congress (SWC). IEEE, 2023: 1-9.
2. <a name="ref2"></a>Zhou Z, Liu C, Yu X, et al. Deep-WiID: WiFi-based contactless human identification via deep learning[C]//2019 IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI). IEEE, 2019: 877-884.
3. <a name="ref3"></a>Wang D, Yang J, Cui W, et al. CAUTION: A Robust WiFi-based human authentication system via few-shot open-set recognition[J]. IEEE Internet of Things Journal, 2022, 9(18): 17323-17333.
4. <a name="ref4"></a>Wang D, Zhou Z, Yu X, et al. CSIID: WiFi-based human identification via deep learning[C]//2019 14th International Conference on Computer Science & Education (ICCSE). IEEE, 2019: 326-330.
6. <a name="ref5"></a>Zhang J, Wei B, Wu F, et al. Gate-ID: WiFi-based human identification irrespective of walking directions in smart home[J]. IEEE Internet of Things Journal, 2020, 8(9): 7610-7624.
7. <a name="ref6"></a>Lin C, Hu J, Sun Y, et al. WiAU: An accurate device-free authentication system with ResNet[C]//2018 15th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON). IEEE, 2018: 1-9.
8. <a name="ref7"></a>Liang Y, Wu W, Li H, et al. WiAi-ID: Wi-Fi-Based Domain Adaptation for Appearance-Independent Passive Person Identification[J]. IEEE Internet of Things Journal, 2023, 11(1): 1012-1027.