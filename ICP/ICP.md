## Learning ICP algorithm
1. Tutorial: https://www.youtube.com/watch?v=OgkH45ofXiw&list=PLdMorpQLjeXmbFaVku4JdjmQByHHqTd1F&index=16
2. Loca and Cost Jacob: https://cgabc.xyz/posts/cfb7b6d6/ (Include the references)
3. Set the jacob size: https://zhuanlan.zhihu.com/p/545458473

## TODO-lists
1. Update to use the ceres Manifold class -> Can we set the Jacobian (e w.r.t. local parameters) directly?
2. Try with using SE(3) as optimisation parameters
3. Update my note and achive following things:
    - Derive the Jacobian of the error function by hand (w.r.t. SE(3) and se(3))
4. Learn:
    - LM method (line searching and trust region method)
    - Adding the rubust kernal - Cauthy losses etc.
