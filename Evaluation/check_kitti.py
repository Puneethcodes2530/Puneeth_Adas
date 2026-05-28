import os

KITTI = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI"

for folder in [
    'data_object_image_2',
    'data_object_label_2',
    'data_object_calib'
]:

    path = os.path.join(KITTI, folder)

    if os.path.exists(path):

        subdirs = os.listdir(path)

        print(f"\n✓ {folder}/")

        for s in subdirs[:3]:

            sub = os.path.join(path, s)

            if os.path.isdir(sub):

                files = os.listdir(sub)

                print(f"    {s}/ → {len(files)} files")

                if files:
                    print(f"       sample: {files[0]}")

    else:

        print(f"✗ {folder} NOT FOUND")