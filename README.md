# Image-Panoramas
The goal of this task is to experiment with image stitching methods.
Given a set of photos, your program should be able to stitch them into a panoramic photo. There are no restrictions regarding the method you use to stitch photos into a panoramic photo. 
# For this project, you can assume the following:
• Your code will need to be able to stitch together anywhere from 2-5 image and you will not know that in advance.  
• You can assume that IF an image is to be part of the panorama, it will overlap at least one other image and by at least 20%.  
• Images that do not overlap with any other image can be ignored.  
• Images will overlap with at most one other image.  
• You are only expected to produce one panorama. For example, if images 1 and 2 overlap, and 3 and 4 overlap, but there is not overlap between 1,2 and 3,4 producing either composite/-panorama is fine.   
# Please keep in mind:
• You will have to determine the spatial arrangement of the images automatically, they can be presented to your code in any order.  
• Although the Figure below shows horizontal panoramas, you five images can be stitched together in any way.  
• While some of the most modern techniques may use a spherical projection for better panoramas, you are free to assume that basic 2D Planer transformations are sufficient for this project.  
