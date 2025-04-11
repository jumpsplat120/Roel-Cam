import streamlink
import cv2
import numpy as np
import imageio
import discord
import asyncio
from discord.ext import tasks, commands
import uuid
from threading import Thread
import os
import copy
import random

#NOTE: CTRL+F for NOTE when looking through code for all the gotcha's and quirks.
#NOTE: Make sure you create a DISCORD_KEY file where you put your discord key.

def streamToURL(url, quality = "best"):
    streams = streamlink.streams(url)
    
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError(f"No stream could be found with the url '{url}'.")

def keypressed(key, timing):
    return  cv2.waitKey(timing) & 0xFF == ord(key)

def createBot():
    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot("!", intents = intents)

    @bot.event
    async def on_ready():
        print(f"We have logged in as {bot.user}")

        bot.forward_channels = []

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if message.content == "Hey Roel, post videos here.":
            if message.author.global_name == "jumpsplat120":
                bot.video_channel = message.channel
                
                await message.channel.send("Okay! I'll post any videos I capture here.")
            else:
                await message.channel.send("Uhm... you're not Mona.")

        if message.content == "Hey Roel, forward videos here.":
            bot.forward_channels.append(message.channel)

            await message.channel.send("Okay! I'll forward any videos I capture here.")

        if message.content == "CAT!":
            await message.channel.send("CATTTTT!")
        elif message.content.lower() == "cat!":
            await message.channel.send("Cat!")
            
        if message.content == "cat":
            await message.channel.send("I demand excitement for kibben >:C")

        
    return bot

""" Obtains image mask
Inputs: 
    fg_mask - foreground mask
    kernel - kernel for Morphological Operations
Outputs: 
    mask - Thresholded mask for moving pixels
"""
def getMotionMask(fg_mask, min_thresh = 0, kernel = np.array((9,9), dtype = np.uint8)):
    _, thresh   = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)
    
    # morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

""" Obtains initial proposed detections from contours discoverd on the
mask. Scores are taken as the bbox area, larger is higher.
Inputs:
    mask - thresholded image mask
    thresh - threshold for contour size
Outputs:
    detectons - array of proposed detection bounding boxes and scores 
                [[x1,y1,x2,y2,s]]
"""
def getContourDetections(mask, thresh = 400):
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    
    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area > thresh: # hyperparameter
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections)

def getBackgroundSubtractor(stype):
    if stype == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
        # backSub.setShadowThreshold(0.75)
    else:
        return cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)

""" Removes all smaller boxes that are contained within larger boxes.
Requires bboxes to be soirted by area (score)
Inputs:
    boxes - array bounding boxes sorted (descending) by area 
            [[x1,y1,x2,y2]]
Outputs:
    keep - indexes of bounding boxes that are not entirely contained 
            in another box
"""
def removeContainedBoundingBoxes(boxes):
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep

"""
Perform non-max suppression on a set of bounding boxes 
and corresponding scores.
Inputs:
    boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    scores: a list of corresponding scores 
    threshold: the IoU (intersection-over-union) threshold for merging bboxes
Outputs:
    boxes - non-max suppressed boxes
"""
def NMS(boxes, scores, threshold=1e-1):
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]
    
    # remove all contained bounding boxes and get ordered index
    order = removeContainedBoundingBoxes(boxes)
    
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    
    return boxes[keep]

""" Main function to get detections via Frame Differencing
Inputs:
    backSub - Background Subtraction Model
    frame - Current BGR Frame
    bbox_thresh - Minimum threshold area for declaring a bounding box
    nms_thresh - IOU threshold for computing Non-Maximal Supression
    kernel - kernel for morphological operations on motion mask
Outputs:
    detections - list with bounding box locations of all detections
        bounding boxes are in the form of: (xmin, ymin, xmax, ymax)
"""
def getDetections(back_sub, frame, bbox_thresh = 100, nms_thresh = 0.1, kernel = np.array((9,9), dtype = np.uint8)):
    # Update Background Model and get foreground mask
    fg_mask = back_sub.apply(frame)
    
    # get clean motion mask
    motion_mask = getMotionMask(fg_mask, kernel = kernel)

    # get initially proposed detections from contours
    detections = getContourDetections(motion_mask, bbox_thresh)
    
    if len(detections) == 0:
        return detections
    
    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # perform Non-Maximal Supression on initial detections
    return NMS(bboxes, scores, nms_thresh)

def drawBoundBoxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
    
    return frame

def getKey():
    file = open("DISCORD_KEY", encoding = "utf-8")	
    key = file.read()

    file.close()

    return key

async def main():
    bot = createBot()
    
    class CatSpotter(commands.Cog):
        def __init__(self, bot, url, quality = "best", fps = 30.0, gif_length = 3.0, gif_max = 300):
            self.stream = cv2.VideoCapture(streamToURL(url, quality))
            self.bot    = bot

            self.detected_motion = False
            self.recording       = False
            self.previous_frame  = None
            self.result          = 0
            self.frame_time      = int((1.0 / fps) * 1000.0)
            self.gif_length      = int(gif_length * fps)
            self.gif_max         = gif_max
            self.frames          = []
            self.rolling_average = []
            self.calculate_bg    = []
            
            self.cat_lines = [
                "Is this a cat maybe?",
                "I defintely saw a cat!",
                "Probably a cat I think!",
                "Look, look, a cat!",
                "I'm thinking cat :3",
                "I do not have a cactus, I have a cat!",
                "I'm 97.425% \sure this is a cat.",
                "Beep boop this a cat.",
                "Cat spotted!",
                "It's giving cat.",
                "ROEL I SEE YOU",
                "Lookie! A cat!",
                "woag is cat",
                "I'm like mostly pretty sure this a cat.",
                "Ignore the haters, I'm the best at noticing cats.",
                "Haha! Cat got you in 4k!",
                "catcatcatcatcat",
                "I don't know what this is at all actually.",
                "Cat! CATTTT",
                "Gatto!",
                "Sowelli!",
                "kibben",
                ":eye: :lips: :eye:"
            ]

            self.bgsub  = getBackgroundSubtractor("MOG2")
            self.kernel = np.array((9,9), dtype=np.uint8)

            #gif_length is the *minimum* length of a gif, but a gif
            #can be longer if motion continues to happen, which will
            #extend the length

            self.manageStream.start()

        def cog_unload(self):
            self.manageStream.cancel()
            
            self.cv2.destroyAllWindows()
            self.stream.release()

        @tasks.loop(seconds = 0)
        async def manageStream(self):
            #Attempt to get frame from the stream
            try:
                _, frame = self.stream.read()
            except:
                print("Failed to read frame.")

            #Do not pass this point unless we have a frame to work with
            if frame is None:
                return

            #The detection code was pulled pretty much wholesale from the
            #following repo:
            #https://github.com/itberrios/CV_projects/blob/main/motion_detection/detection_with_background_subtraction.ipynb
            #There's a corresponding Medium article, but TLDR
            #Get a bunch of frames to determine the "background", then
            #anything on the background is now "object". Get bounding
            #boxes of object. Motion detected. Fine tuning can be done
            #to account for shadows and the like, but I've left it mostly
            #default, except to increase the bounding box threshold
            #so that it doesn't find 10 pixels and assume that the noise
            #is motion.
            detections = getDetections(self.bgsub, 
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                bbox_thresh = 10000, 
                                nms_thresh = 1e-2, 
                                kernel = self.kernel)
            
            #If there's been a dramatic change in the scene, then set detected_motion
            #to gif_length. NOTE: MAGIC NUMBER
            if len(detections) > 2:
                print(f"Motion! Things detected: {len(detections)}")
            
                self.detected_motion = self.gif_length

            #If we haven't started recording, then set the recording flag. We do this
            #to avoid "starting" multiple times.
            if self.detected_motion == self.gif_length and not self.recording:
                print("Beginning recording.")

                self.recording = True
            
            #If the detected_motion counter is above 0, countdown. When we hit
            #zero, we're done recording. This value can continuously get reset
            #if there continues to be motion, which is why we don't just check
            #the framecount; a video can be anywhere from 10 - 30 seconds.
            if self.detected_motion > 0:
                self.detected_motion = self.detected_motion - 1

            #If we're recording, add the color corrected frame to the frames array.
            if self.recording:
                self.frames.append(drawBoundBoxes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detections))
            
            #Max amount of frames a vid can have, to avoid vids being too big.
            #Discord limit.
            if len(self.frames) > self.gif_max:
                self.detected_motion = 0

            #If detected_motion has counted all the way down, then we've
            #finished grabbing all interesting frames. Run the createVideo
            #method without waiting for it to finish, because there *might*
            #be more stuff we want to capture. Also, we copy the frames
            #so that newly added frames don't fuck with the list while we're
            #converting to video.
            if self.detected_motion == 0 and self.recording:
                #This loop is the discord.py event loop. There may be another
                #way to get this via the lib itself, but eh.
                loop = asyncio.get_running_loop()
                
                loop.create_task(self.createVideo(copy.deepcopy(self.frames)))
                
                self.frames    = []
                self.recording = False

        async def createVideo(self, frames):
            #Create a randomly named file. We don't use TemporaryNamedFile
            #to avoid permission issues (possibly just a windows thing, could
            #probably be fixed by running the Terminal in Admin mode but w/e)
            filename = str(uuid.uuid4()) + ".mp4"

            #spawn a seperate thread when doing imageIO conversion to an mp4
            #by default it's blocking, which then causes the Discord bot to
            #freak out cause it gets backed up. So a seperate thread lets
            #it process without that being a problem. 
            thread = Thread(
                target = imageio.v3.imwrite,
                args = (filename, frames),
                kwargs = {
                    "extension": ".mp4",
                    "plugin": "FFMPEG",
                    })
            
            thread.start()
            
            #Wait until thread is finished
            while thread.is_alive():                
                await asyncio.sleep(0)

            #Arbitrarily wait an amount of time so that the file is
            #ready to be accessed (checking with os.access no work :<)
            #TODO: This is the dumbest thing, and could fail randomly
            #It would be nice to figure out a more accurate way to verify
            #That the file can be sent before sending it.
            await asyncio.sleep(10)
            await self.sendVideo(filename)
            
            #NOTE: For reasons unknown, this only works sometimes? Like,
            #most videos get removed, but not all of them. Probably another
            #"you can access this but not really" issue, where python has it
            #for discord.File, and then sends it, and then needs to release it
            #but it hasn't yet, so we can't os.remove. Idk.
            os.remove(filename)

        async def sendVideo(self, path):
            print(f"Sending {path}...")

            file = discord.File(path)

            if self.bot.video_channel is None:
                #NOTE: Magic channel ID. #general in Nova's Books
                await self.bot.get_channel("1307628817032155172").send("Mona! You forgot to set a channel for me! I have cat videos I wanna send you!")

                print("No video channel is set!")
            else:
                #Send the channel to a single channel, and forward everywhere else
                #cause we're being nice to the corporation's bandwidth
                message = await self.bot.video_channel.send(random.choice(self.cat_lines), file = file)
                message_ref = discord.MessageReference.from_message(message)

                message_ref.type = discord.MessageReferenceType.forward

                for i in range(len(self.bot.forward_channels)):
                    await self.bot.forward_channels[i].send(reference = message_ref)
                
                print("Sent.")

    await bot.add_cog(CatSpotter(bot, "https://www.twitch.tv/blahajparty"))
    await bot.start(getKey())

asyncio.run(main())