# Extreme-Weather-Event-Detection
A program to detect impacts of extreme weather events like floodings or wildfires for risk management and emergency response. 

<br>
In the first programm version, it will feature a Flood Detection. It will be possible for the User to input Coordinates of Interest (e.g. point or bounding box) via command line arguments or by providing a JSON file with filter options. The programm will scan the ESA Copernicus Hub server for recently uploaded Satellite Imagery of the area and download it. When the download is finished, a Segmented Image into flooded / not-flooded areas will be outputted. 
<br> <br>
The segmentation will be performed by a Neural Network created with *Tensorflow*. The model creation will be done seperately with Python in a Jupyter Notebook, whereas the Inference will be performed by another Python program.
<br> <br>
Below is a poster, which shows the different components of the project and their interaction.
<br> <br>

![Flood-Prediction-Poster](https://user-images.githubusercontent.com/78846141/143872262-ac91c6f8-fb7d-4438-89de-4b46798a74d7.png)
