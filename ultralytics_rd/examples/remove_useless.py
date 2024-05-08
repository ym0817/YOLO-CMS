import os, cv2

def rm_faces(path):

    mode_list = ['train']
    cnt = 0
    for mode in mode_list:
        images_path = os.path.join(path,mode,'images')
        labels_path = images_path.replace('images', 'labels')
        for label in os.listdir(labels_path):
            label_file = os.path.join(labels_path, label)
            stats = os.stat(label_file)
            img_file = os.path.join(labels_path, label).replace('labels', 'images').replace('.txt', '.jpg')
            im = cv2.imread(img_file)
            # cv2.imshow('img', im)
            # cv2.waitKey(0)
            if stats.st_size == 0 or im is None:
                cnt += 1
                os.remove(label_file)
                os.remove(img_file)
                # img_file = os.path.join(labels_path, label).replace('labels','images').replace('.txt','.jpg')
                # im = cv2.imread(img_file)
                # cv2.imshow('img',im)
                # cv2.waitKey(0)
            # size = os.path.getsize('path\to\file\filename.ext')
            # print(size)
            # print(stats.st_size)
        print(mode, ' remove num : ',cnt)

def change_id(path):

    mode_list = ['Train','Val']
    cnt = 0
    for mode in mode_list:
        images_path = os.path.join(path, mode, 'images')
        labels_path = images_path.replace('images', 'labels')
        new_labels_path = images_path.replace('images', 'labels_new')
        if not os.path.exists(new_labels_path):
            os.makedirs(new_labels_path)

        for label in os.listdir(labels_path):
            label_file = os.path.join(labels_path, label)
            stats = os.stat(label_file)
            img_file = os.path.join(labels_path, label).replace('labels', 'images').replace('.txt', '.jpg')
            im = cv2.imread(img_file)
            # cv2.imshow('img', im)
            # cv2.waitKey(0)
            if stats.st_size == 0 or im is None:
                cnt += 1
                os.remove(label_file)
                os.remove(img_file)
            else:
                lines = open(label_file, 'r').readlines()
                fn = open(label_file.replace('labels', 'labels_new'),'w')
                for line in lines:
                    new_line = line.strip().split()[1:]
                    fn.write('1 ' + ' '.join(new_line) + '\n')
                fn.close()


        print(mode, ' remove num : ', cnt)




if __name__ == '__main__':

    # path = './Face'
    path = './DVR'
    # rm_faces(path)

    change_id(path)









