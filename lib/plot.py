import matplotlib.pyplot as plt


def plot_scatter(y):
    """
    Plot scatter

    Arguments:
    y -- data points

    Return:
    plot saved into a .png file in folder plots
    """

    # Create a list of x-axis values
    x = list(range(1, len(y) + 1))

    # Visualize the scatter plot
    plt.scatter(x, y)

    # Add labels and title
    plt.xlabel("Interations")
    plt.ylabel("Cost")
    plt.title("Costs")

    plt.savefig("plots/costs.png")


def plot_image(x, t):
    """
    Plot data

    Arguments:
    x -- image
    t - title of plot

    Return:
    plot saved into a .png file in folder plots
    """

    # Visualize the data:
    plt.imshow(x)

    plt.title(t)

    plt.savefig("plots/image.png")
