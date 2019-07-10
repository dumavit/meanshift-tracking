#### Tested MeanShift and CamShift tracking algorithms on different datasets


Usage:
    
    `python meanshift.py --dataset dataset  --roi x,y,width,height --method method_name`
    
   

Meanshift works good when the target doesn't change its color and there are no other objects with the same color as target.
Meanshift is invariant to object rotations, because they use color histograms to find objects.
Both Meanshift and Camshift  work pretty bad when background has the same color as tracking objects.
Meanshift has constant bounding box size, thus it's not good at tracking increasing and decreasing objects.

Different masks can be used to filter hsv image.
If tracking object is very bright - it's better to use mask with high HSV value filter.
If tracking object mainly consists of one color - it's better to filter other colors in hue filter.
