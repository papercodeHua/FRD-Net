## model

* code for "FRD-Net: A full-resolution dilated convolutional network for retinal vessel segmentation"
* Huang Hua, Zhenhong Shang, and Chunhui Yu
* Paper:(https://opg.optica.org/viewmedia.cfm?uri=boe-15-5-3344&seq=0)
* Corresponding author email:(szh@kust.edu.cn)
* other author email:(huahuang@tju.edu.cn)

## FRD-NET model structure
The network architecture of FRD-Net,include Dilated Residual Module and Multi-scale feature fusion module
<img src="https://github.com/papercodeHua/FRD-Net/blob/main/images/structure1.svg" width="600">

## experimental result
Visualization comparison of the method with other methods on DRIVE、CHASE_DB1、STARE
<img src="https://github.com/papercodeHua/FRD-Net/blob/main/images/result1.svg" width="600">
</br>
Visualization comparison of the method with other methods on HRF
<img src="https://github.com/papercodeHua/FRD-Net/blob/main/images/result2.svg" width="600">

## How to train

* Before training, please pay attention to the learning rate decay measurement, crop_Size, calculation of normalized
  mean and standard deviation, batch_ Size is 4 and 2

## Citations

*Huang, Hua, Zhenhong Shang, and Chunhui Yu. "FRD-Net: a full-resolution dilated convolution network for retinal vessel segmentation." Biomedical Optics Express 15.5 (2024): 3344-3365.
