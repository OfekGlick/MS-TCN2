import cv2
import os
from tqdm import tqdm


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return


video_names = [f'P0{i}_balloon1' for i in [16, 20, 22]]
gestures = {"no gesture": "G0",
            "needle passing": "G1",
            "pull the suture": "G2",
            "Instrument tie": "G3",
            "Lay the knot": "G4",
            "Cut the suture": "G5"}
gestures = {v: k for k, v in gestures.items()}
for vid in video_names:
    image_folder = f'/datashare/APAS/frames/{vid}_side'
    label_path = f"/home/student/FinalProject/Ofek/results/exp7/fold0/Weighted/test/{vid}"
    video_name = f'LabeledVideos/{vid}.mp4'
    with open(label_path, 'r') as f:
        labels = f.readlines()[1].split()
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    for image, label in tqdm(zip(images, labels)):
        image = cv2.imread(os.path.join(image_folder, image))
        draw_text(image, gestures[label])
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
