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