import os
import pickle
# from subprocess import call
import cv2
import numpy as np
import h5py

data_dir = '/files/yxue/research/allstate/data/CMU_MOSI'

video_dir     = data_dir+'/Video/Segmented'
video_processed_dir = data_dir+'/Video/frames5_224'

transcript_dir            = data_dir+'/Transcript/Segmented'
transcript_processed_dir  = data_dir+'/Transcript/trs'

resize_height = 224
resize_width = 224

def process_video(vdir, odir, fps):
    nframes = {}

    # FNULL = open(os.devnull, 'w')
    for root, dirs, files in os.walk(vdir):
        for f in files:
            video_path = os.path.join(root,f)

            video_name = f.split('.')[0]

            images_dir = os.path.join(odir+'_%sfps'%fps, video_name)
            if not os.path.exists(images_dir):
            	os.makedirs(images_dir)

            # resized_images_dir = os.path.join(odir+'_resized', video_name)
            # if not os.path.exists(resized_images_dir):
            # 	os.makedirs(resized_images_dir)

            # num_frames = len(os.listdir(resized_images_dir))
            # if num_frames >= 30:
            # 	nframes[video_name] = num_frames
            # 	continue

            # print(resized_images_dir, num_frames)

            capture = cv2.VideoCapture(video_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = int(capture.get(cv2.CAP_PROP_FPS))

            # original fps=30
            # set extract fps=3
            EXTRACT_FREQUENCY = original_fps//fps
            while frame_count // EXTRACT_FREQUENCY < 5:
            	EXTRACT_FREQUENCY = EXTRACT_FREQUENCY // 2

            count = 0
            i = 0
            retaining = True

            while (count < frame_count and retaining):
                retaining, frame = capture.read()
                if frame is None:
                    continue

                if count % EXTRACT_FREQUENCY == 0:
                    # if (frame_height != resize_height) or (frame_width != resize_width):
                    #     resized_frame = cv2.resize(frame, (resize_width, resize_height))
                    resized_frame = cv2.resize(frame, (resize_width, resize_height))
                    
                    cv2.imwrite(filename=os.path.join(
                        images_dir, '0000{}.jpg'.format(str(i))), 
                    img=frame)

                    # cv2.imwrite(filename=os.path.join(
                    #     resized_images_dir, '0000{}.jpg'.format(str(i))), 
                    # img=resized_frame)
                    i += 1
                count += 1

            # Release the VideoCapture once it is no longer needed
            capture.release()

            nframes[video_name] = {'total frames': frame_count, 'extracted frames': i, 'duration (s)': frame_count/original_fps}

            print(video_name, nframes[video_name])
            # call(['ffmpeg',
            #     '-i', video_path,
            #     '-r', '1',
            #     '-qscale:v', '10',
            #     '-s', '640*360',
            #     '-threads', '4',
            #     images_dir + '/%04d.jpg'],
            #     stdout=FNULL,
            #     stderr=FNULL)

            # nframes[f.split('.')[0]]=len([name for name in os.listdir(images_dir) if os.path.isfile(name)])

    with open(odir+'_%sfps_frame_count.dict.pkl'%fps, 'wb') as f:
        pickle.dump(nframes, f)

def process_transcript(trsdir, odir):
    trs_dict = {}
    for root, dirs, files in os.walk(trsdir):
        for f in files:
            trs_path = os.path.join(root,f)
            video_name = f.split('.')[0]

            with open(trs_path, 'r', encoding="utf-8") as f:
                raw_trs = f.readlines() 

            for l in raw_trs:
                seg_id = l.split('_')[0]
                seg_name = '%s_%s'%(video_name, seg_id)
                trs_dict[seg_name] = l.split('_')[-1].strip()
                
    with open(odir+'.dict.pkl', 'wb') as f:
        pickle.dump(trs_dict, f)

def split(seg_names, odir):
    np.random.seed(918)
    split_dict = {}
    
    val_size = 220
    test_size = 220

    np.random.shuffle(seg_names)

    for video_name in seg_names[:test_size]:
        split_dict[video_name] = 'test'

    for video_name in seg_names[test_size:(test_size+val_size)]:
        split_dict[video_name] = 'val'

    for video_name in seg_names[(test_size+val_size):]:
        split_dict[video_name] = 'train'

    with open(os.path.join(odir,'split.dict.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)

def process_labels(data_dir):
    label_dict = {}
    data = h5py.File(data_dir+'/CMU_MOSI_Opinion_Labels.csd','r')['Opinion Segment Labels']['data']
    for k in data:
        labels = np.array(data[k]['features']).reshape(-1)
        for i in range(len(labels)):
            label_dict['%s_%s'%(k,i+1)] = labels[i]
    with open(os.path.join(data_dir,'labels.dict.pkl'), 'wb') as f:
        pickle.dump(label_dict, f)

fps=1
process_video(video_dir, video_processed_dir, fps)
process_transcript(transcript_dir, transcript_processed_dir)

with open(transcript_processed_dir+'.dict.pkl', 'rb') as f:
    d = pickle.load(f)
seg_names = list(d.keys())
split(seg_names, data_dir)
process_labels(data_dir)

# process_video(video_dir, video_processed_dir, 3)
# process_video(video_dir, video_processed_dir, 6)
# process_video(video_dir, video_processed_dir, 10)
# process_video(test_video_dir, test_processed_dir)

# process_qa(train_qa_dir, train_qa_processed_dir)
# process_qa(test_qa_dir, test_qa_processed_dir)
# split(train_video_dir, data_dir+'/train')

# test video length
# (array([35, 43, 44, 46, 47, 48, 49, 50, 52, 56, 57, 58, 59, 60]), array([ 1,  1,  1,  2,  3, 19,  3, 10,  1,  3,  1, 11,  6, 38]))
# train video length
# (array([  2,  18,  26,  27,  30,  32,  34,  36,  37,  39,  40,  42,  43,
#         44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
#         57,  58,  59,  60, 117]), array([  1,   1,   1,   2,   4,   3,   1,   2,   3,   1,   5,   1,   3,
#          4,  15,  22,  59, 117,  17,  42,   3,   2,   6,   9,  20,  32,
#         49, 100, 169, 320,   1]))