import glob
import os
import csv
import matplotlib.pyplot as plt

cd = "logs/"
type_datasets = ['pcam','tcga']

for csv_file in glob.glob(os.path.join(cd, "*.csv")):
    accuracies = []
    train_losses = []
    val_losses = []

    with open(csv_file, newline='') as f:
        reader = csv.reader(f,delimiter=',')
        column_titles = next(reader)

        dataset = 'unknown'
        for ds in type_datasets:
            if any(ds.lower() in col.lower() for col in column_titles):
                dataset = ds
                break

        for row in reader:
            accuracies.append(row[3])
            train_losses.append(row[1])
            val_losses.append(row[2])

    accuracies = accuracies[1:(len(accuracies))]
    train_losses = train_losses[1:(len(train_losses))]
    val_losses = val_losses[1:(len(val_losses))]

    for i in range(len(accuracies)):
        train_losses[i] = float(train_losses[i])
        val_losses[i] = float(val_losses[i])
        accuracies[i] = float(accuracies[i])*100
        accuracies[i] = round(accuracies[i],1)
    epochs = range(len(accuracies))

    if dataset == 'tcga':
        plt.plot(epochs,accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy over epochs ({dataset})')
        plt.grid()
        plt.savefig(f'plots/accuracy_{len(epochs)}_epochs_{dataset}.png',dpi=120)
        plt.show()

    else: #pcam
        plt.plot(epochs,accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('F1-score')
        plt.title(f'F1-score over epochs ({dataset})')
        plt.grid()
        plt.savefig(f'plots/f1score_{len(epochs)}_epochs_{dataset}.png',dpi=120)
        plt.show()

    plt.plot(epochs,train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.title(f'Train loss over epochs ({dataset})')
    plt.grid()
    plt.savefig(f'plots/train_loss_{len(epochs)}_epochs_{dataset}.png',dpi=120)
    plt.show()

    plt.plot(epochs,val_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Val loss')
    plt.title(f'Val loss over epochs ({dataset})')
    plt.grid()
    plt.savefig(f'plots/val_loss_{len(epochs)}_epochs_{dataset}.png',dpi=120)
    plt.show()