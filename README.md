pls, orz...

```
@misc{wang2025solvingminmaxmultipletraveling,
      title={Solving the Min-Max Multiple Traveling Salesmen Problem via Learning-Based Path Generation and Optimal Splitting}, 
      author={Wen Wang and Xiangchen Wu and Liang Wang and Hao Hu and Xianping Tao and Linghao Zhang},
      year={2025},
      eprint={2508.17087},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.17087}, 
}
``


1. compile the splitting algorithm

```bash
cd splitting_solver/
mkdir build
cd build
cmake ..
make
mv *.so ../../
```

2. evaluate

```bash
python3 evaluate.py
```

3. training

```bash
python3 train.py --problem mtsp --epochs 100 --nodes 100 --agents_min 2 --agents_max 10 --folder tsp-100-2-10 --aug 16 --batch 512 --history 0 --seed 1234 --rescale 1 --div 4 --port 32423
```

4. pretrained models

The pretrained models could be downloaded from `https://box.nju.edu.cn/f/6a10cf8c189c4198b42a/`, please decompress all files into the `models` folder before evaluation.
