# Require 
tensorflow2.3+    
opencv   
scipy  



# Keras-PoseNet
**This is an implementation for Keras of the [PoseNet architecture](http://mi.eng.cam.ac.uk/projects/relocalisation/)**

As described in the ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]



## Getting Started

 * Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)

 * run train.py and modify the path of the dataset rootpath in line 12 of helper.py   
 * The PoseNet model is defined in the posenet.py file

 * To run:
   * Extract the King's College dataset to wherever you prefer
   * Extract the starting and trained weights to the same location as test.py and train.py
   * Update the dataset path on line 9 in helper.py
   * If you want to retrain, simply run train.py (note this will take a long time)
   * If you just want to test, simply run test.py , may be need to modify by your self 
