
### simple binary matrix library in c++

It only have 2D matrix (vector<vector<int>>) and 1D vector<int>

#### functions
- nc_init.cpp: eye/arange/zeros_vec/zeros_mat
- nc_rand.cpp: generate_gaussian_noise (mean, stddev) 
- numcpp.cpp : kron/setdiff1d/bin_mat_sum/bin_mat_mul/bin_to_dec
  - dec_to_bin/ mat2comp_vec/comp_vec2mat/ swap_row/ add_row

### exmaples 
<!-- - (for more example, see z_example/ folder) -->
- there is --opt 0~5
####  init 2D matrix : zeros, eye

```sh
mkdir build 
cd build
cmake ..
cmake --build . --config Release  
./main -o 1

#* if you are lazy, run.py is a script to compile and run the cpp files 
pip install pyrallis  #python command line library
python run.py  --delete False --opt 1
```

```
zeros vector (3)
0 0 0 
zeros matrix (3,3)
[[0,0,0,],
 [0,0,0,],
 [0,0,0,]]
eye
[[1,0,0,],
 [0,1,0,],
 [0,0,1,]]
arange(1,5,1)
1 2 3 4     
```

#### 1D vector and 2d matrix 
- binary multiplication of 2 matrix, setdiff1d
```sh
python run.py --delete False --opt 2 
```

```
F4
[[1,0,0,0,],
 [1,1,0,0,],
 [1,0,1,0,],
 [1,1,1,1,]]
sum(e1)=4
(F4@e1)%2=
[[1,0,0,0,],
 [1,1,0,0,],
 [1,0,1,0,],
 [1,1,1,1,]]
h stack(A,B,C)
[[1,0,0,0,1,0,0,0,1,0,0,0,],
 [1,1,0,0,0,1,0,0,1,1,0,0,],
 [1,0,1,0,0,0,1,0,1,0,1,0,],
 [1,1,1,1,0,0,0,1,1,1,1,1,]]
setdiff1d (arange(0,5), [1,3])
0 2 4 
e1
[[1,0,0,0,],
 [0,1,0,0,],
 [0,0,1,0,],
 [0,0,0,1,]]
swap e1 row0 row1
[[0,1,0,0,],
 [1,0,0,0,],
 [0,0,1,0,],
 [0,0,0,1,]]
bin add e1 row 1 += row2
[[0,1,0,0,],
 [1,0,1,0,],
 [0,0,1,0,],
 [0,0,0,1,]]
---bin_to_dec
0 1 1 0 1 
--->13
back to vec
0 1 1 0 1 
F4 to compressed int vector
1 3 5 15 
back to mat
[[1,0,0,0,],
 [1,1,0,0,],
 [1,0,1,0,],
 [1,1,1,1,]]
```

#### Ndim Tensor (work in progress)
```sh
python run.py --opt 3 --delete False
```
- for a tensor (dimision > 2)  
- currently we only support print , indexing, slice, reshape 
- tensor (matrix) multiplication and other functions (zeros,randn)...may be added in the future 
- The tensor implementation is based on [pynorch](https://github.com/lucasdelimanogueira/PyNorch)  
- for more information please read `z_exmaple/..Tensor.md`