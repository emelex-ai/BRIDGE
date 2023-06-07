import numpy as np

#seed = 1337
#nb_samples = 1000
#train_test_split = 0.9
#random.seed(seed)
#np.random.seed(seed)

##torch.manual_seed(seed)  
##torch.cuda.manual_seed_all(seed)
##torch.backends.cudnn.deterministic = True
##torch.backends.cudnn.benchmark = False

#  Parameters specific to the main code

def compute_L2_norm(mat):
    norm = np.linalg.norm(mat)
    return norm

if __name__ == '__main__':   # pragma: no cover
    mat = np.random.rand(4, 5)
    norm = compute_L2_norm(mat)
    print(norm)
