
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v1<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<th valign="bottom">DenseCL<br/>top-1 acc.</th>
<th valign="bottom">RegionCL-D<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">53 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">67.5&plusmn;0.1</td>
<td align="center"> 63.8 &plusmn;0.1</td>
<td align="center"> -- &plusmn;0.1</td>
</tr>
<tr><td align="left">ResNet-101</td>
<td align="center">200</td>
<td align="center">--</td>
<td align="center">--</td>
<td align="center">--</td>
<td align="center"> 65.4 &plusmn;0.1</td>
<td align="center"> -- &plusmn;0.1</td>
</tr>
</tbody></table>

My platform is like this: 

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.80.02
* cuda 10.2/11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.11.0

