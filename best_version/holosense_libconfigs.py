import cv2
import os
import math
class configs: # configurations for the camera, display, and user
    
    # Default configurations.
    # Currently configured for display and webcam of Dell Precision 5570 Mobile Workstation

    # Camera ID and resolution
    cameraname = "/dev/video2"
    camw = 1280 
    camh = 720
    FOV = 80
    htan = math.tan(FOV * math.pi / 360) * camw/((camh ** 2 + camw ** 2) ** 0.5)
    vtan = math.tan(FOV * math.pi / 360) * camh/((camh ** 2 + camw ** 2) ** 0.5)
    # These values can be found on the manufacturer's datasheet of your camera.
    # This assumes that your camera has no distortion of the image plane
    # because these types of distortion are not common in laptop webcams
    
    htan = 0 # horizontal FOV in degrees.
    vtan = 0 # vertical field of view

    # Display resolution.
    dispw = 1920 
    disph = 1080 

    # Measurements of real world space. All are in inches.

    screen_diagonal = 15.6 #in inches
    camera_y_offset = 4.25 # vertical distance between the center of the screen and the lens of the webcam.
    camera_x_offset = 0 # horizontal distance parallel to screen
    camera_z_offset = 0 # horizontal distance normal to screen

    # Measurements of your face. See reference image for details.

    eedist = 4
    endist = 3

    def configcamera(self, cap):

        # Here is where you configure the video capture settings.
        # Edit configurations like exposure based on your setup
        os.environ["DISPLAY_DIAGONAL"] = str(self.screen_diagonal)
        os.environ["DISPLAY_WIDTH"] = str(self.dispw)
        os.environ["DISPLAY_HEIGHT"] = str(self.disph)

        
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camh)
        ret_val, frame = cap.read()
        #print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0))
        #print(cap.set(cv2.CAP_PROP_EXPOSURE , 0))        
    
    # DATA LOGGING
    
    
