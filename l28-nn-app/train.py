#%%from fastai import *
from fastai.vision.all import *

path = untar_data(URLs.BIWI_HEAD_POSE)


def img2pose(x):
    return Path(f"{str(x)[:-7]}pose.txt")


cal = np.genfromtxt(path / "01" / "rgb.cal", skip_footer=6)


def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return tensor([c1, c2])


biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == "13"),
    batch_tfms=[
        *aug_transforms(size=(240, 320)),
        Normalize.from_stats(*imagenet_stats),
    ],
)

dls = biwi.dataloaders(path)

learn = cnn_learner(dls, resnet18, y_range=(-1, 1))

lr = 1e-2
learn.fine_tune(3, lr)
#%%
learn.export()

#%%
model_path = Path()
learn_inf = load_learner(model_path / "export.pkl")

img_path = path / "03" / "frame_00003_rgb.jpg"
prediction = learn_inf.predict(img_path)

print(prediction)
print(get_ctr(img_path))
