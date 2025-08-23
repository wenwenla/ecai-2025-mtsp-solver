import argparse
import os
import numpy as np
import pickle


def generate_mtsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2))

def generate_mpdp_data(dataset_size, pdp_size):
    tmp = list(zip(np.random.uniform(size=(dataset_size, 2)),  # Depot location
                np.random.uniform(size=(dataset_size, pdp_size, 2))))
    
    result = []
    for i in range(dataset_size):
        depot = tmp[i][0]
        cities = tmp[i][1]
        comb = np.concatenate([depot[np.newaxis, :], cities], axis=0)

        result.append(comb)
    result = np.stack(result, 0)
    print(result[-1])
    return result
    return list(zip(np.random.uniform(size=(dataset_size, 2)),  # Depot location
                np.random.uniform(size=(dataset_size, pdp_size, 2))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="problem", type=str, required=True)
    parser.add_argument("--cities", help="number of cities", type=int, required=True)
    parser.add_argument("--seed", help="random seed", type=int, required=True)
    parser.add_argument("--folder", help="save dir", type=str, default="testdata")

    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.problem == "mtsp":
        data = generate_mtsp_data(100, args.cities)
    elif args.problem == "mpdp":
        data = generate_mpdp_data(100, args.cities)

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    fn = os.path.join(args.folder, f'{args.problem}_{args.cities}_{args.seed}.pkl')
    with open(fn, 'wb') as fout:
        pickle.dump(data, fout)
    print(data[-1])