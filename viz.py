import pyaudio
from sklearn.preprocessing import normalize
import audioop
import numpy as np
import cv2

import librosa

import time


# get input to how many seconds the beat should consider, and if to run fullscreen
b_secs = int(input('How many seconds should the live-tempo reading consider?: '))
fullscreen = str(input('Should this run in full-screen? (y/n): '))
if fullscreen.lower() == 'y':
    fullscreen = True

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 0.5
   
# Blue color in BGR 
color = (127, 127, 255) 
  
# Line thickness of 2 px 
thickness = 2


# Setup channel info
#FORMAT = pyaudio.paInt16 # data type formate
FORMAT = pyaudio.paInt16 # data type formate

CHANNELS = 2 # Adjust to your number of channels
RATE = 22050 # Sample Rate
CHUNK = int(32*2 * 18*2 * 1) # Block Size (32x18x3)
#RECORD_SECONDS = 5 # Record time
#WAVE_OUTPUT_FILENAME = "file.wav"

# Startup pyaudio instance
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print ("Listening")

from keyboard import is_pressed as key

ticks = 0
canvas = np.ones((450,800,3), dtype=np.uint8)

base = np.ones((450,800,3), dtype=np.uint8) * 255


points = np.ones((2,3))

if fullscreen == True:
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow("window")
'''
bg = cv2.imread('nuri.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.resize(bg, (800,450))
'''
order_types = ['F', 'F']
order_ticks = 0

#cat_array = cv2.imread('kitty_use.jpg')

images = []

import os
pics = os.listdir('pics')
for pic in pics:
    images.append( cv2.cvtColor(cv2.imread(os.path.join('pics',pic)), cv2.COLOR_BGR2GRAY) )
    
#cat_array = cv2.cvtColor(cv2.imread('kitty_use.jpg'), cv2.COLOR_BGR2GRAY)
#print(cat_array.shape)

carlton_visibility = 0.0

frac = 0.7

cat_chance = 20

pic_index = 2

vol_frac = 0.1

#mic_pow = 1.0

#chunk_size = 32*2*18*2
#audio backlog duration is X seconds
backlog_duration_in_seconds = b_secs
audio_backlog = []
backlog_array = None
#audio_backlog = np.zeros((32*2*18*2*10*backlog_duration_in_seconds))
#backlog_ticker = 0

def handle_audio_backlog():
    global audio_backlog, backlog_array
    backlog_array = audio_backlog[0]
    for idc in range(len(audio_backlog) - 1):
        backlog_array = np.append(backlog_array, audio_backlog[idc+1])

#tempo, beats = librosa.beat.beat_track(y=unchanged_song_array, sr=song_rate)
tempo, beats = 0, None


def handle_tempo():
    global tempo, start_time
    
    per_second_beats = tempo / 60
    
    if per_second_beats > 0:
        beat_location_in_second = 1 / per_second_beats
    else:
        beat_location_in_second = 1
        
    remainder = (beat_location_in_second - ((time.time() - start_time) % beat_location_in_second))
    #print(tempo)
    #print(remainder)
    #return beat_status
    if remainder < 0.115:
        beat_status = True
    else:
        beat_status = False
    return beat_status
    
# THE CARLTON CONTROLLER!!!
# stat is either left, mid, or right
carlton_stat = 'mid'

car_l2 = cv2.cvtColor(cv2.resize(cv2.imread('carlton/l2.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_l1 = cv2.cvtColor(cv2.resize(cv2.imread('carlton/l1.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_mid_l = cv2.cvtColor(cv2.resize(cv2.imread('carlton/idle_left.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_mid_m = cv2.cvtColor(cv2.resize(cv2.imread('carlton/idle_mid.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_mid_r = cv2.cvtColor(cv2.resize(cv2.imread('carlton/idle_right.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_r1 = cv2.cvtColor(cv2.resize(cv2.imread('carlton/r1.png'), (246,350)), cv2.COLOR_BGR2GRAY)
car_r2 = cv2.cvtColor(cv2.resize(cv2.imread('carlton/r2.png'), (246,350)), cv2.COLOR_BGR2GRAY)

# initialize carlton
shown_carlton = car_mid_m

#carlton_ticker = 0

carlton_in_swing = False

swing_dir = 'out'


#,'mic_pow: z/x'
info_list = ['canv_frac: w/s','im_chance: c/v','im_idc: o/p(r/t)','vol_frac: l/k','carlton: n/b']


def carlton_control(beat_stat):
    global carlton_stat, carlton_in_swing, shown_carlton, swing_dir, rms
    ''' can happen every other frame is using carlton ticker, for now will happen every frame '''
    
    if (beat_stat == False) and (carlton_in_swing == False):
        # brign back to idle state
        if np.array_equal(shown_carlton, car_r2):
            shown_carlton = car_r1
        elif np.array_equal(shown_carlton, car_r1):
            shown_carlton = car_mid_r
        elif np.array_equal(shown_carlton, car_l2):
            shown_carlton = car_l1
        elif np.array_equal(shown_carlton, car_l1):
            shown_carlton = car_mid_l
        # cycle between the mid idle animations
        if (np.array_equal(shown_carlton, car_mid_r)) or (np.array_equal(shown_carlton, car_mid_l)):
            shown_carlton = car_mid_m
            carlton_stat = 'mid'
        if np.array_equal(shown_carlton, car_mid_m):
            # flip a coin for direction
            coin = np.random.randint(0, 2)
            if coin == 0:
                shown_carlton = car_mid_l
                carlton_stat = 'left'
            elif coin == 1:
                shown_carlton = car_mid_r
                carlton_stat = 'right'
    
    if beat_stat == True and rms > 0.01:
        # check if in swing
        if carlton_in_swing == False:
            # check direction to swing at
            if carlton_stat == 'mid':
                coin = np.random.randint(0, 2)
                if coin == 0:
                    shown_carlton = car_mid_l
                    carlton_stat = 'left'
                elif coin == 1:
                    shown_carlton = car_mid_r
                    carlton_stat = 'right'
            elif carlton_stat == 'left':
                shown_carlton = car_l1
                carlton_in_swing = True
                swing_dir = 'out'
            elif carlton_stat == 'right':
                shown_carlton = car_r1
                carlton_in_swing = True
                swing_dir = 'out'
    
    # if in swing, continue swing until middle
    if carlton_in_swing == True:
        # check if swing going out or in
        if swing_dir == 'out':
            if carlton_stat == 'left':
                if np.array_equal(shown_carlton, car_l1):
                    shown_carlton = car_l2
                    swing_dir = 'in'
            elif carlton_stat == 'right':
                if np.array_equal(shown_carlton, car_r1):
                    shown_carlton = car_r2
                    swing_dir = 'in'
        elif swing_dir == 'in':
            if carlton_stat == 'left':
                if np.array_equal(shown_carlton, car_l2):
                    shown_carlton = car_l1
                elif np.array_equal(shown_carlton, car_l1):
                    shown_carlton = car_mid_l
                    carlton_in_swing = False
            elif carlton_stat == 'right':
                if np.array_equal(shown_carlton, car_r2):
                    shown_carlton = car_r1
                elif np.array_equal(shown_carlton, car_r1):
                    shown_carlton = car_mid_r
                    carlton_in_swing = False
                    
start_time = time.time()
while True:
    
    order_ticks += 1
    if order_ticks > 1:
        order_ticks = 0
    
    data = stream.read(CHUNK)
    rms = audioop.rms(data, 2) 
    # the rms is probably up to 32767
    rms = rms / 32767
    
    # now i can calculate the audio loudness, as it would move between 0 and 1
    decoded = np.fromstring(data, 'Float32')
    
    
    
    # account for nan
    nan_locations = np.isnan(decoded)
    decoded[nan_locations] = 0.
        
    inf_locations = np.where(decoded > 32767)
    decoded[inf_locations] = 32767.

    inf_locations = np.isposinf(decoded > 32767)
    decoded[inf_locations] = 32767.
    
    #positive inf already changed, leaving only negative inf
    inf_locations = np.where(decoded < -32767)
    decoded[inf_locations] = -32767.
        
    inf_locations = np.isinf(decoded > 32767)
    decoded[inf_locations] = 32767.
        
    # normalize
    normal_decoded = normalize(decoded.reshape(-1,1), norm='max', axis=0)
    
    inf_locations = np.isposinf(normal_decoded)
    normal_decoded[inf_locations] = 1.
    
    inf_locations = np.isinf(normal_decoded)
    normal_decoded[inf_locations] = 0.
    '''
    neg_locations = np.where(normal_decoded < 0.)
    print(neg_locations)
    normal_decoded[nan_locations] = 0.
    '''
    
    #nan_locations = np.isnan(normal_decoded)
    #print(nan_locations)
    
    #print(normal_decoded.max())
    #print(normal_decoded.min())
    
    
    # store array in backlog for tempo getting
    audio_backlog.append(normal_decoded)
    if len(audio_backlog) > (backlog_duration_in_seconds*10):
        del audio_backlog[0]
    
    handle_audio_backlog()
    
    # Get the tempo!

    #tempo, beats = librosa.beat.beat_track(y = np.squeeze(backlog_array), sr = RATE)
    try:
        tempo, beats = librosa.beat.beat_track(y = backlog_array, sr = RATE)
    except:
        print('NO BEAT CAN DO BABY')
        pass
        
    b_stat = handle_tempo()
    
    
    
    '''if b_stat == True:
        cv2.circle(canvas, (400, 225), 50, (255,255,255), 1)'''
    
    # BACK TO NORMAL TEMPO
    decoded = normal_decoded
    # account for inf and -inf
    neg_locations = np.where(decoded < 0.)
    decoded[nan_locations] = 0.
    pos_locations = np.where(decoded > 1.)
    decoded[nan_locations] = 1.
    # account for nan (again)
    nan_locations = np.isnan(decoded)
    decoded[nan_locations] = 0.
    #print(nan_locations)
    inf_locations = np.isinf(decoded)
    #print(inf_locations)
    decoded[inf_locations] = 0.
    
    inf_locations = np.isinf(decoded)
    #print(inf_locations)
    
    
    decoded = np.reshape(decoded, (int(18*2),int(32*2)), order=order_types[order_ticks])
    
    decoded[decoded < 0.] = 0.
    decoded[decoded > 1.] = 1.
    
    
    #print(decoded.shape)
    
    #print(decoded)
    
    
    #, cv2.INTER_NEAREST
    #canvas[:,:,ticks] = cv2.resize(255 * decoded.astype(np.uint8), (800,450), interpolation = cv2.INTER_NEAREST)
    canvas[:,:,ticks] = (cv2.resize(255 * frac * decoded.astype(np.uint8), (800,450), interpolation = cv2.INTER_NEAREST) + (canvas[:,:,ticks] * (1 - frac)))
    
    # CONTROL AND DRAW CARLTON
    carlton_control(b_stat)
    # carlton is 350 by 500
    # make carlton mask
    used_carlton = shown_carlton.copy() * carlton_visibility
    
    carlton_mask = np.where(used_carlton > 0.25)
    # draw carlton on canvas
    canvas[50 : 400, 277 : 523, ticks][carlton_mask] = used_carlton[carlton_mask]
    if ticks == 2:
        canvas[50 : 400, 277 : 523, 0][carlton_mask] = (used_carlton[carlton_mask] / 1.5)
    else:
        canvas[50 : 400, 277 : 523, ticks + 1][carlton_mask] = (used_carlton[carlton_mask] / 1.5)
    
    
    pos_locations = np.argwhere(cv2.cvtColor(cv2.resize(canvas,(64,36)), cv2.COLOR_BGR2GRAY) == 1.0)
    #decoded[nan_locations] = 1.
    #print(pos_locations.shape)
    for idc in range(pos_locations.shape[0]):
        coin_flip = np.random.randint(0,cat_chance)
        if coin_flip == 0:
            pt = pos_locations[idc,:]
            #n_pt = pos_locations[idc + 1, :] * 12.5
            pt = pt * 12.5
            #print(pt)
            #cv2.circle(canvas, (int(pt[1] + 6.25), int(pt[0] + 6.25)), np.random.randint(5,30), (255,255,255), 1)
            
            # decide on size of pic used
            ran_size = np.random.randint(10,70)
            # make sure it's always an even number
            if ran_size % 2 != 0:
                ran_size += 1
            
            if (pic_index != 'R') and (pic_index != 'T'):
                img_use = images[int(round(pic_index))]
            elif pic_index == 'R':
                img_use = images[np.random.randint(0, len(images))]
            elif pic_index == 'T':
                cv2.circle(canvas, (int(pt[1] + 6.25), int(pt[0] + 6.25)), np.random.randint(5,50), (255,255,255), 1)
            
            if pic_index != 'T':
                img_now = cv2.resize(img_use, (ran_size, ran_size))
                #print(ran_size)
                #print(int(ran_size/2))
                try:
                    img_mask = np.where(img_now != 0)
                    #canvas[int(pt[0] - (ran_size/2)) : int(pt[0] + (ran_size/2)), int(pt[1] - (ran_size/2)) : int(pt[1] + (ran_size/2)) , ticks][img_mask] = (canvas[int(pt[0] - (ran_size/2)) : int(pt[0] + (ran_size/2)), int(pt[1] - (ran_size/2)) : int(pt[1] + (ran_size/2)) , ticks][img_mask]/3) + (img_now[img_mask] * rms)
                    canvas[int(pt[0] - (ran_size/2)) : int(pt[0] + (ran_size/2)), int(pt[1] - (ran_size/2)) : int(pt[1] + (ran_size/2)) , ticks][img_mask] = (canvas[int(pt[0] - (ran_size/2)) : int(pt[0] + (ran_size/2)), int(pt[1] - (ran_size/2)) : int(pt[1] + (ran_size/2)) , ticks][img_mask] * (1-rms)) + (img_now[img_mask] * (rms + vol_frac))
                    
                except:
                    pass
       
        
    '''
    # get points from b, g and r (try)
    try:
        M = cv2.moments(canvas[:,:,ticks])
        cY = int(M["m01"] / M["m00"])
        cX = int(M["m10"] / M["m00"])
        
        ydif = 450 - cY - 32
        xdif = 800 - cX - 32
        
        randX = np.random.randint(low = -min(xdif, cX), high = min(xdif, cX))
        randY = np.random.randint(low = -min(ydif, cY), high = min(ydif, cY))
         
        points[:,ticks] = np.array([cX + randX, cY + randY])
        #points[:,ticks] = np.array([cX, cY])
    
    except:
        pass
    
    for p in range(3):
        cv2.circle(canvas, (int(points[0,p]), int(points[1,p])), np.random.randint(20,100), (255,255,255), np.random.randint(1,3))
        cv2.circle(canvas, (int(points[0,p]), int(points[1,p])), np.random.randint(1,10), (255,255,255), np.random.randint(1,3))
        #print(canvas[ int(points[1,p]) : int(points[1,p]) + 24, int(points[0,p]) : int(points[0,p]) + 24 ,p].shape)
        
        #canvas[ int(points[1,p]) : int(points[1,p]) + 32, int(points[0,p]) : int(points[0,p]) + 32 ,p]= cat_array + #canvas[ int(points[1,p]) : int(points[1,p]) + 32, int(points[0,p]) : int(points[0,p]) + 32 ,p]
        
    
    cv2.line(canvas, (int(points[0,0]), int(points[1,0])), (int(points[0,1]), int(points[1,1])), (200,255,200), np.random.randint(1,3))
    cv2.line(canvas, (int(points[0,1]), int(points[1,1])), (int(points[0,2]), int(points[1,2])), (200,255,200), np.random.randint(1,3))
    cv2.line(canvas, (int(points[0,2]), int(points[1,2])), (int(points[0,0]), int(points[1,0])), (200,255,200), np.random.randint(1,3))
    '''

    
    ticks += 1
    if ticks > 2:
        ticks = 0
    
    #' : ' + str(round(mic_pow,2)) + 
    cv2.putText(canvas, str(round(frac,2)) + ' : ' + str(cat_chance) + ' : ' + str(pic_index) + ' : ' + str(round(vol_frac,2)) + ' : ' + str(carlton_visibility), (20,20), font, fontScale, color, thickness, cv2.LINE_AA) 
    
    cv2.putText(canvas, str(round(rms,2)) + ' : ' + str(round(tempo,2)) + ' : ' + str(b_stat), (20,50), font, fontScale, (255,165,165), thickness, cv2.LINE_AA) 
    
    
    for idc in range(len(info_list)):
        itm = info_list[idc]
        cv2.putText(canvas, itm, (20, 70 + (idc * 20)), font, fontScale, (190,255,190), 1, cv2.LINE_AA) 
    
    cv2.imshow("window", canvas)
    k = cv2.waitKey(1)
    if k == 27:
        break
    
    if key('w'):
        frac += 0.01
    elif key('s'):
        frac -= 0.01
    if frac < 0.1:
        frac = 0.1
    if frac > 1.:
        frac = 1.
    
    if key('c'):
        cat_chance -= 1
    elif key('v'):
        cat_chance += 1
    if cat_chance < 1:
        cat_chance = 1
    if cat_chance > 100:
        cat_chance = 100
    
    if key('o'):
        if (pic_index != 'R') and (pic_index != 'T'):
            pic_index -= 0.25
        else:
            pic_index == 0
    elif key('p'):
        if (pic_index != 'R') and (pic_index != 'T'):
            pic_index += 0.25
        else:
            pic_index = 0
    elif key('r'):
        pic_index = 'R'
    elif key('t'):
        pic_index = 'T'
        
    if (pic_index != 'R') and (pic_index != 'T') and (pic_index > (len(images) - 1.)):
        pic_index = len(images) - 1
    if (pic_index != 'R') and (pic_index != 'T') and (pic_index < 0.):
        pic_index = 0
    
    if key('l'):
        vol_frac += 0.01
    elif key('k'):
        vol_frac -= 0.01
    if vol_frac < 0.0:
        vol_frac = 0.0
    if vol_frac > 1.:
        vol_frac = 1.
    
    if key('n'):
        carlton_visibility += 1
    elif key('b'):
        carlton_visibility -= 1
    if carlton_visibility < 0:
        carlton_visibility = 0
    if carlton_visibility > 1:
        carlton_visibility = 1
    '''
    if key('x'):
        mic_pow += 0.01
    elif key('z'):
        mic_pow -= 0.01
    if mic_pow < 0.0:
        mic_pow = 0.0
    if mic_pow > 1.:
        mic_pow = 1.
    '''
# Stop Recording
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()