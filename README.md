
## experiment results

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">IN-linear</th>
<th valign="bottom">IN-finetune</th>
<th valign="bottom">coco-bbox</th>
<th valign="bottom">coco-segm</th>
<th valign="bottom">cityscapes</th>
<th valign="bottom">link</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">mocov2 r50</a></td>
<td align="center">67.36</td>
<td align="center">77.07</td>
<td align="center">38.68</td>
<td align="center">33.88</td>
<td align="center">77.88</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/selfsup-model_1.tar">model_1</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2207.08220">+fast-moco</a></td>
<td align="center">70.83</td>
<td align="center">77.16</td>
<td align="center">39.30</td>
<td align="center">34.38</td>
<td align="center">77.94</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/selfsup-model_2.tar">model_2</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2111.12309">+cutmix</a></td>
<td align="center">70.86</td>
<td align="center">77.22</td>
<td align="center">39.17</td>
<td align="center">34.31</td>
<td align="center">78.41</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/selfsup-model_3.tar">model_3</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2011.09157">+dense</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center"></td>
<td align="center"><a href="https://github.com/CoinCheung/DenseCL/releases/download/v0.0.1/regioncl_r101_checkpoint_0199.pth.tar">model_4</a></td>
</tr>
</tbody></table>

Notes:   
    **IN-linear** means linear evaluation on imagenet.   
    **IN-finetune** means finetune on imagenet.   
    **coco-bbox** means object detection on coco.   
    **coco-segm** means instance segmentation on coco.  
    **cityscapes** means semantic segmentation on cityscapes.   


## training platform: 

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.80.02
* cuda 11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.12.0




## raw results
model_1:   
    linear:  
        Acc@1 67.416 Acc@5 87.872  
        Acc@1 67.312 Acc@5 87.886  
        Acc@1 67.320 Acc@5 87.812  
        Acc@1 67.404 Acc@5 87.866  
    finetune:  
        Acc@1 77.252 Acc@5 93.598  
        Acc@1 76.902 Acc@5 93.478  
        Acc@1 77.028 Acc@5 93.550  
        Acc@1 77.114 Acc@5 93.582  
    coco:  
        bbox: 38.9088,58.6155,42.1195,22.5249,43.4853,53.2623  
        segm: 34.1413,55.4126,36.3194,15.2867,37.2440,51.9860  
        bbox: 38.1508,57.6392,41.1611,20.7222,42.8992,51.8489  
        segm: 33.4272,54.5981,35.3582,14.3102,36.7986,50.7192  
        bbox: 38.7340,58.1209,42.1626,22.4818,43.5662,52.4255  
        segm: 33.9001,54.8911,36.1609,15.3182,37.3886,50.5798  
        bbox: 38.9785,58.5592,42.1268,22.3429,43.7718,52.9737  
        segm: 34.0710,55.2887,36.2458,15.4479,37.4157,50.8065  
    deeplab:  
        78.2688,58.6464,90.2149,77.6179  
        77.7682,57.8095,90.2813,77.4702  
        78.1918,58.4643,90.2833,77.6382  
        77.3166,58.1030,90.2396,77.6255  
   
model_2:   
    linear:  
        Acc@1 70.778 Acc@5 89.818  
        Acc@1 70.858 Acc@5 89.918  
        Acc@1 70.868 Acc@5 89.872  
        Acc@1 70.854 Acc@5 89.946  
    finetune:  
        Acc@1 77.244 Acc@5 93.468  
        Acc@1 77.214 Acc@5 93.490  
        Acc@1 77.122 Acc@5 93.524  
        Acc@1 77.096 Acc@5 93.560  
    coco:  
        bbox: 38.8650,58.7779,41.7332,22.4263,43.9251,51.6782  
        segm: 33.9814,55.3545,35.9995,15.5256,37.7407,50.4110  
        bbox: 39.6814,59.4448,42.9916,22.3039,44.6076,53.6434  
        segm: 34.6912,56.1206,36.8248,15.7709,38.3276,51.9963  
        bbox: 39.4916,59.3736,42.8254,23.1930,44.4368,52.5870  
        segm: 34.5428,56.0413,36.6260,15.7507,38.4682,51.1193  
        bbox: 39.1963,59.0715,42.3853,22.1433,44.3828,52.2242  
        segm: 34.3486,55.7250,36.5795,15.4824,38.0844,50.9105  
    deeplab:  
        78.1100,59.0041,90.3268,78.1409   
        78.2087,59.1489,90.3703,78.0988  
        77.5934,58.0122,90.3006,77.9366  
        77.8860,58.6174,90.3806,78.2742  
    

model_3:   
    linear:  
        Acc@1 70.876 Acc@5 89.834  
        Acc@1 70.772 Acc@5 89.876   
        Acc@1 70.886 Acc@5 89.764  
        Acc@1 70.958 Acc@5 89.742  
    finetune:  
        Acc@1 77.240 Acc@5 93.550   
        Acc@1 77.326 Acc@5 93.462  
        Acc@1 77.256 Acc@5 93.568  
        Acc@1 77.094 Acc@5 93.512  
    coco:  
        bbox: 39.3460,59.0546,42.2851,22.9140,44.4155,53.3382  
        segm: 34.5142,55.9223,36.6815,15.7571,38.1347,52.0390  
        bbox: 39.0491,58.8558,41.9208,22.2061,44.0689,52.6811  
        segm: 34.1807,55.5958,36.2895,15.4523,37.7644,51.8671  
        bbox: 39.2131,59.0032,42.3043,22.1091,44.6531,52.9608  
        segm: 34.3523,55.6958,36.7397,15.3001,38.2401,51.3491  
        bbox: 39.1080,58.9902,42.0726,22.3417,44.6007,52.6574  
        segm: 34.2092,55.5251,36.2403,15.4809,37.8173,51.5715  
    deeplab:  
        78.7164,59.3104,90.5327,78.5289  
        78.4461,58.4039,90.4018,78.0944  
        78.7509,59.2488,90.5220,78.0358  
        77.7645,58.0034,90.4262,78.1178  


model_4: 
