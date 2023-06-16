# 3D Pose Estimation of Bearded Dragons for Visual Exploration Research
This was the final project in my bachelor's degree. It was awarded as one of the best projects in both The Blavatnik School of Computer Science (2020-2022 Outstanding Project Competition) and the Sagol School of Neuroscience (2021-2022 Project Presentation Competition). The project was in the fields of Image Processing & Computer Vision where we estimated the 3D pose and location of free-to-move animals using Linear Algebra and Deep Learning.

This project aims to build a research tool that produces a 3D gaze trace over time of the dragons while they exhibit a more natural behaviour when compared to their behaviour when  measured with today’s eye-tracking tools. This new system will allow acquisition of a complete visual exploration behaviour profile of the animal, by the construction of three stages: (1) locating the eyes of the animal in space, (2) modelling the angle of view through changes in the pupil, and (3) connecting all the information together to build an accurate gaze vector for each eye.
The code appears here focuses on solving the first stage of this process.
### Contents:
1. **Project Booklet.pdf** - the project paper contains the project background, description of the physical system (arena's cameras, assembly components, etc.), review of literature, methods (video detection & tracking, cameras' calibration - undistortion, 3D positioning, comibinig the cameras and filtering), results and more.
2. **pose_analysis.ipynb** - the main flow of the system, using all the other files in this repository.
3. **TrainNetwork_VideoAnalysis.ipynb** - the 2D-detection network training notebook using DeepLabCut.
![Bearded Dragons](https://github.com/noyshabtay/CS-Final-Project/blob/main/readme_dragons_pic.png)
