# Automated Board Defect Detection using Deep Learning 

- An interface to carry out defect detection on a PCB indicated through color-coded bounding boxes. 

- The YOLOv8 object detection architecture was utilized to build the model to predict bounding boxes and give the appropriate classification.

- The GUI of the application was built using the Tkinter library. Running the `pcbagui.py` file on an environment with package support will open the GUI in a new window.

- Various other object detection architectures were tested for preciseness (more about which can be found in `Final_Project_report.pdf`) and it was concluded that the YOLOv8 architecture performed most optimally. 
