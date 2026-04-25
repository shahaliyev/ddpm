import matplotlib.pyplot as plt

def show(img_tensor):
    img = img_tensor.squeeze().permute(1,2,0)
    plt.imshow(img)
    plt.show()
