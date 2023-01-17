
## experiment results

Each model is train for 200 epoch.  

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
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/selfsup-model_1.tar">link</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2207.08220">+fast-moco</a></td>
<td align="center">70.83</td>
<td align="center">77.16</td>
<td align="center">39.30</td>
<td align="center">34.38</td>
<td align="center">77.94</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/selfsup-model_2.tar">link</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2111.12309">+cutmix</a></td>
<td align="center">71.32</td>
<td align="center">77.15</td>
<td align="center">39.41</td>
<td align="center">34.47</td>
<td align="center">78.63</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/r50_checkpoint_0199_mocov2_fastm_cutmix.pth.tar">link</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/1710.09412">+mixup</a></td>
<td align="center">70.42</td>
<td align="center">77.28</td>
<td align="center">39.46</td>
<td align="center">34.56</td>
<td align="center">78.54</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/r50_checkpoint_0199_mocov2_fastm_cutmix_mixup.pth.tar">link</a></td>
</tr>

<tr><td align="left"><a href="https://arxiv.org/abs/2011.09157">+dense</a></td>
<td align="center">68.79</td>
<td align="center">77.28</td>
<td align="center">40.00</td>
<td align="center">34.81</td>
<td align="center">78.69</td>
<td align="center"><a href="https://github.com/CoinCheung/SelfSup/releases/download/0.0.0/r50_checkpoint_0199_mocov2_fastm_cutmix_mixup_dense.pth.tar">link</a></td>
</tr>
</tbody></table>


Notes:   
&#8195;&#8195;**IN-linear:**  linear evaluation on imagenet.   
&#8195;&#8195;**IN-finetune:**  finetune on imagenet.   
&#8195;&#8195;**coco-bbox:**  object detection on coco.   
&#8195;&#8195;**coco-segm:**  instance segmentation on coco.  
&#8195;&#8195;**cityscapes:**  semantic segmentation on cityscapes.   
&nbsp;


## training platform: 

* ubuntu 18.04
* 32 nvidia Tesla T4 gpu, driver 450.80.02
* cuda 11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.12.0




## raw results
Each experiment is done 4 times, and above result in the table is the mean of the 4 results.  


mocov2:   
&#8195;linear:  
&#8195;&#8195;Acc@1 67.416 Acc@5 87.872  
&#8195;&#8195;Acc@1 67.312 Acc@5 87.886  
&#8195;&#8195;Acc@1 67.320 Acc@5 87.812  
&#8195;&#8195;Acc@1 67.404 Acc@5 87.866  
&#8195;finetune:  
&#8195;&#8195;Acc@1 77.252 Acc@5 93.598  
&#8195;&#8195;Acc@1 76.902 Acc@5 93.478  
&#8195;&#8195;Acc@1 77.028 Acc@5 93.550  
&#8195;&#8195;Acc@1 77.114 Acc@5 93.582  
&#8195;coco:  
&#8195;&#8195;bbox: 38.9088,58.6155,42.1195,22.5249,43.4853,53.2623  
&#8195;&#8195;segm: 34.1413,55.4126,36.3194,15.2867,37.2440,51.9860  
&#8195;&#8195;bbox: 38.1508,57.6392,41.1611,20.7222,42.8992,51.8489  
&#8195;&#8195;segm: 33.4272,54.5981,35.3582,14.3102,36.7986,50.7192  
&#8195;&#8195;bbox: 38.7340,58.1209,42.1626,22.4818,43.5662,52.4255  
&#8195;&#8195;segm: 33.9001,54.8911,36.1609,15.3182,37.3886,50.5798  
&#8195;&#8195;bbox: 38.9785,58.5592,42.1268,22.3429,43.7718,52.9737  
&#8195;&#8195;segm: 34.0710,55.2887,36.2458,15.4479,37.4157,50.8065  
&#8195;deeplab:  
&#8195;&#8195;78.2688,58.6464,90.2149,77.6179  
&#8195;&#8195;77.7682,57.8095,90.2813,77.4702  
&#8195;&#8195;78.1918,58.4643,90.2833,77.6382  
&#8195;&#8195;77.3166,58.1030,90.2396,77.6255  
   
+fast-moco:   
&#8195;linear:  
&#8195;&#8195;Acc@1 70.778 Acc@5 89.818  
&#8195;&#8195;Acc@1 70.858 Acc@5 89.918  
&#8195;&#8195;Acc@1 70.868 Acc@5 89.872  
&#8195;&#8195;Acc@1 70.854 Acc@5 89.946  
&#8195;finetune:  
&#8195;&#8195;Acc@1 77.244 Acc@5 93.468  
&#8195;&#8195;Acc@1 77.214 Acc@5 93.490  
&#8195;&#8195;Acc@1 77.122 Acc@5 93.524  
&#8195;&#8195;Acc@1 77.096 Acc@5 93.560  
&#8195;coco:  
&#8195;&#8195;bbox: 38.8650,58.7779,41.7332,22.4263,43.9251,51.6782  
&#8195;&#8195;segm: 33.9814,55.3545,35.9995,15.5256,37.7407,50.4110  
&#8195;&#8195;bbox: 39.6814,59.4448,42.9916,22.3039,44.6076,53.6434  
&#8195;&#8195;segm: 34.6912,56.1206,36.8248,15.7709,38.3276,51.9963  
&#8195;&#8195;bbox: 39.4916,59.3736,42.8254,23.1930,44.4368,52.5870  
&#8195;&#8195;segm: 34.5428,56.0413,36.6260,15.7507,38.4682,51.1193  
&#8195;&#8195;bbox: 39.1963,59.0715,42.3853,22.1433,44.3828,52.2242  
&#8195;&#8195;segm: 34.3486,55.7250,36.5795,15.4824,38.0844,50.9105  
&#8195;deeplab:  
&#8195;&#8195;78.1100,59.0041,90.3268,78.1409   
&#8195;&#8195;78.2087,59.1489,90.3703,78.0988  
&#8195;&#8195;77.5934,58.0122,90.3006,77.9366  
&#8195;&#8195;77.8860,58.6174,90.3806,78.2742  
    

+cutmix:   
&#8195;linear:  
&#8195;&#8195;Acc@1 71.328 Acc@5 90.156  
&#8195;&#8195;Acc@1 71.304 Acc@5 90.140   
&#8195;&#8195;Acc@1 71.420 Acc@5 90.122  
&#8195;&#8195;Acc@1 71.244 Acc@5 90.138  
&#8195;finetune:  
&#8195;&#8195;Acc@1 77.144 Acc@5 93.610   
&#8195;&#8195;Acc@1 77.012 Acc@5 93.440  
&#8195;&#8195;Acc@1 77.284 Acc@5 93.570  
&#8195;&#8195;Acc@1 77.208 Acc@5 93.564  
&#8195;coco:  
&#8195;&#8195;bbox: 39.1084,59.1479,42.2791,22.4279,44.4237,53.2122  
&#8195;&#8195;segm: 34.2483,55.8341,36.5099,15.1468,38.1949,51.5365  
&#8195;&#8195;bbox: 39.5533,59.3614,42.9080,22.7960,44.7712,53.9648  
&#8195;&#8195;segm: 34.5953,55.9939,36.8683,15.2826,38.3724,52.1124  
&#8195;&#8195;bbox: 39.4069,59.3092,42.6750,23.1086,44.6547,53.2479  
&#8195;&#8195;segm: 34.5314,55.9698,36.7320,16.6169,38.6138,51.2430  
&#8195;&#8195;bbox: 39.5974,59.5217,42.5368,23.4857,45.2232,53.7143  
&#8195;&#8195;segm: 34.5555,56.2276,36.6526,16.6227,38.6538,52.0167  
&#8195;deeplab:  
&#8195;&#8195;78.6545,58.8794,90.4519,78.2854  
&#8195;&#8195;78.5601,59.4348,90.4387,78.0768  
&#8195;&#8195;78.3252,59.1720,90.4460,78.2306  
&#8195;&#8195;78.9993,59.0939,90.5852,78.4380  


+mixup:  
&#8195;linear:  
&#8195;&#8195;Acc@1 70.426 Acc@5 89.920  
&#8195;&#8195;Acc@1 70.502 Acc@5 89.952  
&#8195;&#8195;Acc@1 70.458 Acc@5 89.982  
&#8195;&#8195;Acc@1 70.292 Acc@5 89.952  
&#8195;finetune:  
&#8195;&#8195;Acc@1 77.232 Acc@5 93.526  
&#8195;&#8195;Acc@1 77.362 Acc@5 93.634  
&#8195;&#8195;Acc@1 77.262 Acc@5 93.532  
&#8195;&#8195;Acc@1 77.280 Acc@5 93.696  
&#8195;coco:  
&#8195;&#8195;bbox: 39.4056,59.0497,42.3511,22.5197,44.3473,53.2681  
&#8195;&#8195;segm: 34.5090,55.8374,36.8892,15.4050,38.3349,51.3551  
&#8195;&#8195;bbox: 39.4914,59.2288,42.7277,21.8810,44.7128,53.6556  
&#8195;&#8195;segm: 34.6709,56.1226,37.1046,15.4873,38.2889,52.5108  
&#8195;&#8195;bbox: 39.4731,59.2949,42.4782,23.3873,44.6141,53.2002  
&#8195;&#8195;segm: 34.6118,56.0852,36.8488,16.4710,38.2405,51.8747  
&#8195;&#8195;bbox: 39.3198,58.9063,42.3989,22.9052,44.1219,53.4169  
&#8195;&#8195;segm: 34.4959,55.8026,36.7658,16.2410,38.2230,52.2665  
&#8195;deeplab:  
&#8195;&#8195;78.7261,58.7874,90.4714,78.3816  
&#8195;&#8195;78.6566,58.2874,90.5395,78.3319  
&#8195;&#8195;78.3647,58.4627,90.4798,78.5871  
&#8195;&#8195;78.4664,58.7298,90.4983,78.3792  


+dense: 
&#8195;linear:
&#8195;&#8195;Acc@1 68.878 Acc@5 88.988
&#8195;&#8195;Acc@1 68.794 Acc@5 88.962
&#8195;&#8195;Acc@1 68.722 Acc@5 88.982
&#8195;&#8195;Acc@1 68.784 Acc@5 88.952
&#8195;finetune:
&#8195;&#8195;Acc@1 77.108 Acc@5 93.560
&#8195;&#8195;Acc@1 77.374 Acc@5 93.696
&#8195;&#8195;Acc@1 77.274 Acc@5 93.560
&#8195;&#8195;Acc@1 77.404 Acc@5 93.630
&#8195;coco:
&#8195;&#8195;bbox: 39.9832,59.9284,43.4874,22.5677,45.1365,54.0637
&#8195;&#8195;segm: 34.9129,56.6404,37.0795,15.4479,38.6572,52.1838
&#8195;&#8195;bbox: 39.7174,59.5656,42.8223,22.8941,45.0750,53.2531
&#8195;&#8195;segm: 34.5298,56.3506,36.8583,15.6896,38.3153,51.9833
&#8195;&#8195;bbox: 40.1698,59.9970,43.3704,23.9472,45.6143,53.9589
&#8195;&#8195;segm: 34.9143,56.3698,37.4172,17.2151,38.8261,52.0241
&#8195;&#8195;bbox: 40.1558,59.7698,43.3668,22.2897,45.6031,54.1690
&#8195;&#8195;segm: 34.9127,56.6232,37.0422,15.5800,38.7052,52.5246
&#8195;deeplab:
&#8195;&#8195;78.4191,58.8781,90.5701,78.7800
&#8195;&#8195;78.7982,59.7372,90.4969,78.4392
&#8195;&#8195;78.9305,59.1239,90.5806,78.7365
&#8195;&#8195;78.6499,58.9398,90.4527,78.1707

