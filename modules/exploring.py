import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def contrast_var_distributions(df, test_vars, segment_var = None):
    """
    Plots a series of histograms to contrast distributions of variables of interest.

    If a segment_var is provided, segments will be plotted in each figure to see how segments differ.
    """
    test_vars = test_vars
    n_test = len(test_vars)
    n_seg = len(df[segment_var].unique()) if segment_var else 1

    fig, ax = plt.subplots(n_test,n_seg,figsize = (8*n_seg,6*n_test))
    for i, col in enumerate(test_vars):
        if segment_var:
            # Get a color mapper object to allow me to color segments differently
            # Will also scale with added/removed segments
            # Color mapper object will have n_seg colors, each of which I can call by index with cmap(), later
            cmap = plt.cm.get_cmap('hsv', n_seg)
            for k, seg in enumerate(df[segment_var].unique()):
                ax[i,k].hist(
                    df[col][df[segment_var] == seg],
                    alpha = 0.5,
                    label = seg,
                    color = cmap(i),
                    edgecolor = 'black'
                )
                ax[i,k].set_title(col + ', Segment ' + str(seg), fontsize = 18)
                ax[i,k].legend()
                # Comma-format y axis ticks. Found here:
                # https://stackoverflow.com/questions/25973581/how-do-i-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib
                ax[i,k].yaxis.set_major_formatter(tck.StrMethodFormatter('{x:,.0f}'))
        else:
            # Add 2 to cmap to avoid multiple charts of same color
            cmap = plt.cm.get_cmap('hsv', n_test + 2)
            ax[i].hist(
                df[col],
                color = cmap(i),
                edgecolor = 'black'
            )
            ax[i].set_title(col, fontsize = 18, pad = 20)
            ax[i].yaxis.set_major_formatter(tck.StrMethodFormatter('{x:,.0f}'))

            fig.tight_layout()

    return fig
