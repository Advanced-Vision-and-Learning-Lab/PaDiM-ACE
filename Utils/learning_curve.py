import matplotlib.pyplot as plt


def learning_curve(num_epochs, train_loss, train_acc, val_loss, val_acc, loss_type):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    fig_name = "learning_curve(loss)"+loss_type+".png"
    plt.savefig(fig_name)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc, label='Validation Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    fig_name = "learning_curve(acc)"+loss_type+".png"
    plt.savefig(fig_name)