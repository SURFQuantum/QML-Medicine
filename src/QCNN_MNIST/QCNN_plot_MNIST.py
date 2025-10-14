from QCNN_model_MNIST import main

import matplotlib.pyplot as plt

test_losses, test_accuracies, n_train, n_test = main()
epochs = range(1, len(test_losses) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(epochs,test_accuracies)
ax1.set_ylabel("Accuracy (%)")
ax1.set_xlabel("Epoch")
ax1.title.set_text(f'Accuracy of {n_train} training images and {n_test} test images')
ax1.grid()

ax2.plot(epochs,test_losses)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epoch")
ax2.title.set_text(f'Loss of {n_train} training images and {n_test} test images')
ax2.grid()

plt.savefig('results/result_plot_QCNN_MNIST.pdf', dpi=300)
plt.tight_layout()
plt.show()