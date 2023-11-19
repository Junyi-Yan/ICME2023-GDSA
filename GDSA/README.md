# GDSA

Code for ICME 2023 paper **"Rethinking graph anomaly detection: A self-supervised Group Discrimination paradigm with Structure-Aware"** https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10220012

![2.png](https://s2.loli.net/2023/11/19/Vb9Q67BclkYGemx.png)

# Overview
Our implementation for GDSA is based on PyTorch. 


#Requirments
This code requires the following:

- Python>=3.8
- PyTorch>=1.8.1
- Numpy>=1.22.4
- Scipy>=1.7.3
- Scikit-learn>=1.0.2
- Networkx>=2.5.1
- Ogb>=1.3.6
- DGL==0.4.3.post2 (Do not use the version which is newer than that!)
#Running the experiments

**Step 1: Anomaly Injection:**
```
python inject_anomaly.py
```

**Step 2: Anomaly Detection:**
```
python main.py
```

# Reference

```
@inproceedings{yan2023rethinking,
  title={Rethinking graph anomaly detection: A self-supervised Group Discrimination paradigm with Structure-Aware},
  author={Yan, Junyi and Zuo, Enguang and Chen, Chen and Chen, Cheng and Zhong, Jie and Li, Tianle and Lv, Xiaoyi},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={2735--2740},
  year={2023},
  organization={IEEE}
}
```

# Contact

If you have any question, please contact yjy@stu.xju.edu.cn (or junyiiyan01@163.com).