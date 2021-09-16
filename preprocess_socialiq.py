import os
import pickle
# from subprocess import call
import cv2

data_dir = '/files/yxue/research/allstate/data/socialiq'

train_video_dir     = data_dir+'_raw/train/vision/raw'
train_processed_dir = data_dir+'/train/vision/videos_1fps_640-360'
test_video_dir      = data_dir+'_raw/test/Segmented/Videos'
test_processed_dir  = data_dir+'/test/vision/videos_1fps_640-360'

train_qa_dir            = data_dir+'_raw/train/qa'
train_qa_processed_dir  = data_dir+'/train/qa'
test_qa_dir             = data_dir+'_raw/test/qa_pairs/qa'
test_qa_processed_dir   = data_dir+'/test/qa'

resize_height = 300
resize_width = 300

def process_video(vdir, odir):
    nframes = {}

    # FNULL = open(os.devnull, 'w')
    for root, dirs, files in os.walk(vdir):
        for f in files:
            video_path = os.path.join(root,f)

            video_name = f.split('.')[0]
            images_dir = os.path.join(odir, video_name)
            os.makedirs(images_dir)
            resized_images_dir = os.path.join(odir+'_resized', video_name)
            os.makedirs(resized_images_dir)

            capture = cv2.VideoCapture(video_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # original fps=30
            # set extract fps=1
            EXTRACT_FREQUENCY = 30

            count = 0
            i = 0
            retaining = True

            while (count < frame_count and retaining):
                retaining, frame = capture.read()
                if frame is None:
                    continue

                if count % EXTRACT_FREQUENCY == 0:
                    if (frame_height != resize_height) or (frame_width != resize_width):
                        resized_frame = cv2.resize(frame, (resize_width, resize_height))
                    
                    cv2.imwrite(filename=os.path.join(
                        images_dir, '0000{}.jpg'.format(str(i))), 
                    img=frame)

                    cv2.imwrite(filename=os.path.join(
                        resized_images_dir, '0000{}.jpg'.format(str(i))), 
                    img=resized_frame)
                    i += 1
                count += 1

            # Release the VideoCapture once it is no longer needed
            capture.release()

            nframes[video_name] = i

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

    with open(odir+'_frame_count.dict.pkl', 'wb') as f:
        pickle.dump(nframes, f)

def process_qa(qadir, odir):
    ques_id = 0
    qa_dict = {}
    for root, dirs, files in os.walk(qadir):
        for f in files:
            qa_path = os.path.join(root,f)

            qa_name = f.split('.')[0]

            video_name = qa_name + '-out'

            with open(qa_path, 'r', encoding="utf-8") as f:
                qa = f.readlines() #[line.decode('utf-8').strip() for line in f.readlines()]

            for l in qa:
                if l[0] == 'q':
                    qa_sample = {
                        'video_name': video_name, 
                        'question': l.split(':')[1].rstrip('\n').lstrip(' ')
                    }
                else:
                    qa_sample_copy = dict(qa_sample)
                    qa_sample_copy['answer'] = l.split(':')[1].rstrip('\n').lstrip(' ')
                    qa_sample_copy['label'] = 0 if l[0] == 'i' else 1
                    qa_dict[ques_id] = qa_sample_copy
                    ques_id += 1
    with open(odir+'.dict.pkl', 'wb') as f:
        pickle.dump(qa_dict, f)

def split(vdir, odir):
    np.random.seed(918)
    video_names = []
    split_dict = {}
    for root, dirs, files in os.walk(vdir):
        for f in files:
            video_path = os.path.join(root,f)
            if f == 'deKPBy_uLkg_trimmed-out.mp4':
                continue

            video_names.append(f.split('.')[0])
    val_size = 100

    np.random.shuffle(video_names)

    for video_name in video_names[:100]:
        split_dict[video_name] = 'val'

    for video_name in video_names[100:]:
        split_dict[video_name] = 'train'

    with open(os.path.join(odir,'split.dict.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)

process_qa(train_qa_dir, train_qa_processed_dir)
process_qa(test_qa_dir, test_qa_processed_dir)
split(train_video_dir, data_dir+'/train')

# process_video(train_video_dir, train_processed_dir)
# process_video(test_video_dir, test_processed_dir)
