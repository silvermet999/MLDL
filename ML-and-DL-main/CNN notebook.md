# CNN 
## use over ANN
many computation

shift in image

## filter map steps
grid (3X3, 4X4..) from original img

multiply with filter

find average (sum of products / 9 for 3X3..)

averages in filter map 

ReLU (non-linearity)

if theres number 1 or ~1 => loopy circle map (eye or nose, the circle in number 9...)

## Features:
auto detection of eyes, legs... using backpropagation + hyperperama for number of filters

## feature extraction
multiply features to get one feature (eyes, nose... to get the head and hands, feet.. for the body)

flatten it in 1D arrays (head and body)

## classification
fully dense connected NN for location flexibility

## Pooling layers
btw each extraction
reduce size of img => reduce Ds, overfitting, tolerant to distortions

### Avg pooling

### Max pooling
max val in each window (grid) in feature map, move in strides

=> helps with position invarient feature detection (detection of 1 or ~1)

![image.png](attachment:image.png)

CNN cant handle rotation and scaling => data aug techn


```python

```
