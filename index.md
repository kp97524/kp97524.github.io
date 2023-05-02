---
layout: default
---



[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

## INTRODUCTION

Computer Vision is used in  various domains like agriculture, automation systems,  autonomous vehicles, robots, security etc.Tasks like object detection and recognition in CV use ML models which are typically intricate and require significant computing power. Some of these applications which were traditionally cloud based, are now moving to edge computing due to benefits like low power consumption, less latency and better security.Our task is focussed on FPGA simulation of models used for object segmentation by analysis of images from video frames taken from various input sources.


## PROBLEM STATEMENT
Using video object segmentation, we can automate tasks which would otherwise require tedious manual effort. Like in the case of surveillance videos, it can be used to alert security personnel to potential threats or suspicious activities in such use cases performing visual object segmentation on cloud adds latency, which is not a viable option. 

In this project, we plan to perform the task of image analysis from video frames taken from various sources such as traffic cameras for identifying objects using effective deep learning models and then implement viable models on an FPGA simulation through VIVADO and metric against task key metrics (like accuracy, mean IoU, Jaccard Score, etc) and constraint performance metrics like resource consumption (e.g. latency, DSP, LUT). Using video object segmentation, we can automate tasks which would otherwise require tedious manual effort. Like in the case of surveillance videos, it can be used to alert security personnel to potential threats or suspicious activities in such use cases performing visual object segmentation on cloud adds latency, which is not a viable option. FPGA implementation of these segmentation algorithms can provide faster results with a reduced power consumption. FPGAs facilitate parallel processing and for tasks like semantic segmentation of videos where large amounts of data need to be processed in parallel.

## PROPOSED WORK

->![Proposed Solution](https://github.com/kp97524/kp97524.github.io/blob/master/PROPOSED_SOLUTION.png?raw=true)


Since there are many different state-of-the-art models available for visual object segmentation, it is important to choose the one that is most suitable for the task at hand. The first step in this plan is to establish a baseline by implementing the most pragmatic state-of-the-art (SOA) model for visual object segmentation. This model will serve as the starting point for further development and optimization towards FPGA implementation. 
The implementation of the current state of the art models was not viable as the current code infrastructure for enabling the conversion of deep learning models for FPGA use have various limitations. Firstly, they do not work with models written as custom classes and the support for pytorch is limited to basic layers and networks only [10]. Hence, most of these models would have to be retrained in tensorflow requiring code conversion and heavy compute resources for model training. 
Alternatively, we implemented the most implemented state of the art model UNET [11] for the task of visual object segmentation. Since the model was lighter in comparison with the other state of the art models, we were able to train the model with our personal compute resources over selected vehicle specific classes from DAVIS 2016 class. Additionally, due to the nascent state of the research in FPGA implementation of ML models, not all the layers are available in the code infrastructure for conversion including Conv2DTranspose. To tackle this issue we have replaced it with a combination of Upsampling 2D and Conv2D layers in the architecture which also proves to be more memory-efficient than using a Conv2D transpose layer. The Conv2D transpose layer requires more memory because it needs to compute a dense matrix multiplication, while UpSampling followed by a Conv2D layer only requires computing a sparse matrix multiplication. For training we used 50 epoch cycles with a total dataset of size 400 car images as seen on the road. These images were split into training and test datasets with a ratio of 9:1 respectively. 
The hls4ml (High-Level Synthesis for Machine Learning) library was used to convert the trained machine learning model into hardware design that could be implemented on FPGA.
The next step is to modify its architecture as necessary so that it is suitable for FPGA implementation. Since FPGA devices have limited resources compared to traditional computing devices, it may be necessary to optimize the architecture of the model to reduce resource utilization and ensure real-time performance. The optimizations of the model like pruning that is used to reduce the complexity and size of the model are still work in progress. The results described in the Evaluation section are the results of a non-optimized model. After optimizing the model, this model would be implemented on FPGA. 


## RELATED WORK

In this paper [1], a Quality-aware Dynamic Memory Network (QDMN) was proposed for video object segmentation. The authors address the mask error accumulation problem, where frames with poor segmentation masks are likely to be memorized by the model. The solution to prevent this issue is to evaluate the segmentation quality of each frame and selectively store the segmented frames. This model achieved state-of-the-art performance on DAVIS and YouTubeVOS benchmark datasets by achieving a mean Intersection over union (IoU) score of 91% on DAVIS 2016 dataset and 82% on YouTube-VOS dataset. The authors in [2] presented a semi-supervised framework for visual object segmentation that was designed with a dynamically scalable architecture for speed-accuracy trade-offs.
Associating Objects with Scalable Transformers approach was used to match and segment multiple objects with online network scalability. This model achieved a state-of-the-art performance of 93% IoU on DAVIS 2016 dataset. In [3], XMEM architecture was proposed to perform video object segmentation on long video datasets. Existing architectures use a single feature memory model that could result in a trade-off between memory consumption and segmentation accuracy. The XMEM model consolidates actively used working memory elements to a long-term memory which avoids the memory explosion problem. This model achieves an IoU score of 92% on the DAVIS 2016 dataset.


## CONTRIBUTION & METHODOLOGY

Since there are many different state-of-the-art models available for visual object segmentation, it is important to choose the one that is most suitable for the task at hand. The first step in this plan is to establish a baseline by implementing the most pragmatic state-of-the-art (SOA) model for visual object segmentation. This model will serve as the starting point for further development and optimization towards FPGA implementation. The implementation of the current state of the art models was not viable as the current code infrastructure for enabling the conversion of deep learning models for FPGA use have various limitations. Firstly, they do not work with models written as custom classes and the support for pytorch is limited to basic layers and networks only [10]. Hence, most of these models would have to be retrained in tensorflow requiring code conversion and heavy compute resources for model training. 

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

## IMPLEMENTATION

Alternatively, we implemented the most implemented state of the art model UNET [11] for the task of visual object segmentation. Since the model was lighter in comparison with the other state of the art models, we were able to train the model with our personal compute resources over selected vehicle specific classes from DAVIS 2016 class. Additionally, due to the nascent state of the research in FPGA implementation of ML models, not all the layers are available in the code infrastructure for conversion including Conv2DTranspose. To tackle this issue we have replaced it with a combination of Upsampling 2D and Conv2D layers in the architecture which also proves to be more memory-efficient than using a Conv2D transpose layer. The Conv2D transpose layer requires more memory because it needs to compute a dense matrix multiplication, while UpSampling followed by a Conv2D layer only requires computing a sparse matrix multiplication. For training we used 50 epoch cycles with a total dataset of size 400 car images as seen on the road. These images were split into training and test datasets with a ratio of 9:1 respectively.

The hls4ml (High-Level Synthesis for Machine Learning) [10] library was used to convert the trained machine learning model into hardware design that could be implemented on FPGA. The HLS model plot is described in Fig 1. The next step is to modify its architecture as necessary so that it is suitable for FPGA implementation. Since FPGA devices have limited resources compared to traditional computing devices, it may be necessary to optimize the architecture of the model to reduce resource utilization and ensure real-time performance. The optimizations of the model like pruning that is used to reduce the complexity and size of the model are still work in progress. The results described in the Evaluation section are the results of a non-optimized model. After optimizing the model, this model would be implemented on FPGA.

->![HLS Model Plot After Conversion](https://github.com/kp97524/kp97524.github.io/blob/master/fig1.png?raw=true) <-
<br>
->*Fig 1: HLS Model Plot After Conversion*<-

### SUB HEADING 1

**********

### SUB HEADING 2

************

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

## RESULTS & EVALUATION

The Tensorflow U-Net model was modified as stated above and this model was used to perform the object segmentation task on DAVIS 2016 dataset. The model was trained on 300 car images from the dataset. The results of evaluating the model on the test set are described in Table 1. The results of some sample predictions are displayed in Fig 2. We observed a continuous decrease in loss and continual increase in the accuracy values during the training process. These results prove that this model outperformed the existing state-of-the-art models for object segmentation tasks.

The baseline model implementation will give us the required metrics to begin our development for optimization. We have considered both software/task performance metrics and computer resource performance metrics for our study. For task performance we shall adopt metrics such as Accuracy, IoU (Intersection over Union) and F1 score. For FPGA implementation, the model was first compiled into a Vivado project using hls4ml by setting the current FPGA to Xilinx xcu250-figd2104-2L-e for virtual simulation. Later, it was synthesized and simulated using Vivado high level synthesis extension (VIVADO HLS) which shall deliver us the usage metrics such as latency, DSP, LUT etc. We believe that the two metrics will present themselves as a trade-off against one another.

| Test Metrics        | Value      |
|:-------------|:------------------|
| Accuracy           | 0.9856 |
| IoU Coefficient | 0.9669   |
| Jaccard Index           | 0.9667  |

*Table 1: Results of Evaluation*


![Sample predictions from images of DAVIS-2016](https://github.com/kp97524/kp97524.github.io/blob/master/fig2.png?raw=true) 
<br>
->*Fig 2: Sample predictions from images of DAVIS-2016*<-

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

## CONCLUSION






## RESOURCES

-- ADD LINKS TO PROJECT PROPOSAL AND MID TERM REPORT

## REFERENCES

1. Liu, Y., Yu, R., Yin, F., Zhao, X., Zhao, W., Xia, W. and Yang, Y., 2022, October.
“Learning quality-aware dynamic memory for video object segmentation.” In
Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October
23–27, 2022, Proceedings, Part XXIX (pp. 468-486). Cham: Springer Nature
Switzerland.
2. Yang, Z., Miao, J., Wang, X., Wei, Y. and Yang, Y., 2022. “Associating objects with
scalable transformers for video object segmentation.” arXiv preprint
arXiv:2203.11442.
3. Cheng, H.K. and Schwing, A.G., 2022, October. “XMem: Long-Term Video Object
Segmentation with an Atkinson-Shiffrin Memory Model.” In Computer Vision–
ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022,
Proceedings, Part XXVIII (pp. 640-658). Cham: Springer Nature Switzerland.

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
