import seaborn as sns
import matplotlib.pyplot as plt


def plot_cm(cm):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='crest')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['no_side_effects', 'had_side_effects'])
    ax.yaxis.set_ticklabels(['had_side_effects', 'no_side_effects'])
    plt.tight_layout()
    plt.show()


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()