## Learning ICP algorithm

This is usefule:
1. https://blog.csdn.net/ljtx200888/article/details/114278683
2. http://ceres-solver.org/nnls_modeling.html#costfunction
3. https://blog.csdn.net/u011092188/article/details/77833022
4. https://github.com/kxhit/semantic-icp/blob/master/semantic_icp/local_parameterization_se3.h
5. https://github.com/chengwei0427/testICP/blob/main/src/lidarOptimization/lidarCeres.cpp

TRY THIS:
https://github.com/ulterzlw/icp_ceres/blob/master/main.cc

Checkout this for the jacobian size
https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

try use Sophus DX_dir to see

Checkout the ceres library on the LocalParam for understanding the size of the jacobian

Checkout that the size of the jacobian for the LocaParam can be different to the size of the jacobian for the CostFunc

Checkout this for understanding more of LIE GROUP with ROBOTICS: http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf

archive above
-----
Loca and Cost Jacob:
https://cgabc.xyz/posts/cfb7b6d6/
