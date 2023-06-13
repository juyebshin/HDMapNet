import os
import sys
import glob
import argparse
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Image sequence to video translation')
    parser.add_argument('--input-dir', type=str, default='runs/distance_vertex/vertex', 
                        help='Directory to sequence of images')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory where output video is saved. default: same as input-dir')
    parser.add_argument('--img-glob', type=str, default='*.png',
                        help='Image extension. default: png')
    parser.add_argument('--fps', type=int, default='10',
                        help='FPS of output video. default: 30fps')
    parser.add_argument('--w', type=int, default=400) # 800 200
    parser.add_argument('--h', type=int, default=200) # 600 196
    parser.add_argument('--scene', action='store_true')
    args = parser.parse_args()
    if args.output_dir == None:
        args.output_dir = args.input_dir
    return args

class VideoStreamer(object):
    def __init__(self, basedir, img_glob, transform=None):
        self.listing = []
        self.i = 0
        self.skip = 1
        self.maxlen = 1000000
        self.transform = transform

        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError(f'No images were found (maybe bad \'--img_glob\' ({img_glob}) parameter?)')
    
    def read_image(self, impath, format='RGB'):
        if format == 'RGB':
            # im = Image.open(impath).convert('RGB')
            im = cv2.imread(impath)
        elif format == 'RGGB':
            # npimg = np.array(Image.open(impath).convert('L'))
            npimg = cv2.imread(impath)
            # Bayer BGGR to RGB
            im = cv2.cvtColor(npimg, code=cv2.COLOR_BAYER_RG2RGB)
        if self.transform is not None:
            im = self.transform(im).unsqueeze(0)
        
        return im
    
    def next_frame(self, format='RGB'):
        if self.i == self.maxlen:
            return (None, None, False)
        
        image_file = self.listing[self.i]
        input_image = self.read_image(image_file, format=format)
        self.i = self.i + 1

        return (input_image, image_file, True)

if __name__ == '__main__':
    args = parse_args()

    vs = VideoStreamer(args.input_dir, img_glob=args.img_glob)

    if not args.scene:
        outname = os.path.join(args.output_dir, 'video.mp4')
        # out = cv2.VideoWriter(outname, cv2.VideoWriter_fourcc(*'DIVX'), 30)
        out = cv2.VideoWriter(filename=outname, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=args.fps, frameSize=(args.w,args.h))

        idx = 0
        while True:
            image, file_name, status = vs.next_frame()
            if status is False:
                break
            if out.isOpened(): # only for front camera
                # print('index: {}'.format(idx))
                # print('image: {}'.format(file_name))
                # print('image size: {}'.format(image.shape))
                out.write(image)
    
        if out.isOpened():
            out.release()
            print('video saved as: {}'.format(outname))
        else:
            print('Failed to save video')
    else: # by scene
        scene_list = []
        out = None
        outname = None
        while True:
            image, file_name, status = vs.next_frame()
            if status is False:
                if len(scene_list) > 0 and out is not None and outname is not None:
                    if out.isOpened():
                        out.release()
                        print('last video saved as: {}'.format(outname))
                print('All videos saved!')
                break
            scene_name = file_name.split('/')[-1].split('_')[0]
            if scene_name not in scene_list:
                if len(scene_list) > 0 and out is not None:
                    if out.isOpened():
                        out.release()
                        print('video saved as: {}'.format(outname))
                        out = None
                scene_list.append(scene_name)
                outname = os.path.join(args.output_dir, f'{scene_name}.mp4')
                out = cv2.VideoWriter(filename=outname, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=args.fps, frameSize=(args.w,args.h))
            
            if out is not None and out.isOpened():
                out.write(image)