
### simple binary matrix library

- it only have 2D matrix (vector<vector<int>>) and 1D vector<int>
- nc_init.cpp: eye/arange/zeros_vec/zeros_mat
- nc_rand.cpp: generate_gaussian_noise (mean, stddev) 
- numcpp.cpp : kron/setdiff1d/bin_mat_sum/bin_mat_mul/bin_to_dec
  - dec_to_bin/ mat2comp_vec/comp_vec2mat/ swap_row/ add_row
```sh
python run.py --opt 0 --delete False
```
this can generate same randn (gauessian) distribution as in python 

```
1608637542
3421126067
0.06456553583016988
0.07325430398666906
```

```sh
python main.py --opt 0
```
```
1608637542
3421126067
0.06456553583016988
0.07325430398666906
```


```sh
python run.py --opt 1 --delete False
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

```sh
python run.py --opt 2 --delete False
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