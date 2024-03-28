# storyboard
This code will help you to create AI voiceover script to video User Upload
# step1
Fist We Initialize the hugging face models by using its serverless endpoint.

# step2
Then we use Tkinter package to upload video from our device

# step 3
Then we captures frames with the help of OpenCV 

# step 4
with the help of captures frames Blip model will generate captions and then Mistral model will take that captions and generate a voiceover script.

# step 5
then we create a pdf using pdf pages and matplotlib, I use 6 frames at one page and individual script on the frame to make it like a proper storyboard like a comic. 

# Note
we use captions for mistral so the script can be relevent to our topic.
