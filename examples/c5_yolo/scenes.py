"""Deterministic, rasterized square/diamond detection scenes and held-out cases."""

import numpy as np

SIZE = 8
CHANNELS = 4
TOP_K = 6
SEED = 53026
CLASS_NAMES = ("square", "diamond")
IMAGE_SCALE = 1 / 16
BOX_SCALE = SCORE_SCALE = 1 / 4
THRESHOLD = 0.50

# Head cells are indexed in fine-then-coarse row-major order. Cell centers
# track the s2/s4 convolution grids, with 0.5 representing a pixel center.
CELLS = tuple((scale,row,col,col*(2<<scale)+.5,row*(2<<scale)+.5)
              for scale,width in ((0,4),(1,2)) for row in range(width) for col in range(width))
GRID = np.array([[x,y,x,y] for _,_,_,x,y in CELLS], np.float32)
INK = np.array([[.95,.06,.03,.03],[.04,.92,.06,.04],
                [.06,.08,.94,.04],[.04,.06,.06,.92]], np.float32)
HELD_OUT = {
    "single": ((5,0,1.00),),
    "multi": ((6,1,1.25),(13,0,1.00),(19,0,1.75)),
    "dense-overlap": ((9,0,1.00),(10,1,1.25),(14,0,1.25),(19,1,2.00)),
    "mixed-scales": ((7,1,1.00),(12,0,1.25),(15,1,1.25),(17,0,1.75)),
}


def render(objects, rng):
    """Draw textured silhouettes, with max intensity preserving overlapping inks."""
    xy = np.arange(SIZE, dtype=np.float32) + .5
    xx, yy = np.meshgrid(xy, xy)
    image = rng.uniform(0, .035, size=(SIZE, SIZE, CHANNELS)).astype(np.float32)
    targets = []
    for index, cls, radius in objects:
        level, _, _, x, y = CELLS[index]
        dx, dy = np.abs(xx-x), np.abs(yy-y)
        distance = np.maximum(dx,dy) if cls == 0 else dx+dy
        silhouette = np.clip(radius + .45 - distance, 0, 1)
        texture = (.24 + .70*np.exp(-1.7*(dx*dx+dy*dy))) * silhouette
        texture *= rng.uniform(.88, 1.08)
        image = np.maximum(image, texture[...,None] * INK[2*level+cls])
        targets.append({"cell":index, "class":cls,
                        "box":[x-radius,y-radius,x+radius,y+radius]})
    return image, targets


def quantized_image(image):
    return np.clip(np.rint(np.asarray(image) / IMAGE_SCALE),-128,127).astype(np.int8)


def training_data(count, seed=SEED):
    rng = np.random.default_rng(seed)
    excluded = {tuple(sorted((i,cls) for i,cls,_ in row)) for row in HELD_OUT.values()}
    images, targets = [], []
    while len(images) < count:
        indices = rng.choice(len(CELLS), size=rng.integers(1,6), replace=False)
        classes = rng.integers(0,2,len(indices))
        layout = tuple(sorted((int(i),int(c)) for i,c in zip(indices,classes,strict=True)))
        if layout in excluded:
            continue
        objects = [(i,c,float(rng.choice((1.0,1.25) if i < 16 else (1.75,2.0)))) for i,c in layout]
        image, target = render(objects, rng)
        images.append(image)
        fields = np.zeros((len(CELLS),8), dtype=np.float32)
        for item in target:
            i = item["cell"]
            fields[i,0] = 1
            fields[i,1+item["class"]] = 1
            fields[i,4:] = np.array(item["box"],np.float32)-GRID[i]
        targets.append(fields)
    return np.stack(images), np.stack(targets)


def held_out():
    rng = np.random.default_rng(SEED+1)
    return {name:render(objects,rng) for name,objects in HELD_OUT.items()}
