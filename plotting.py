from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text


def reg_mean_plot(
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
):
    """
    Plots mean matching figure for a set of specific genes.
    """

    import seaborn as sns

    sns.set()
    sns.set(color_codes=True)

    condition_key = "condition"

    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
        y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
            x_diff, y_diff
        )
        if verbose:
            print("top_100 DEGs mean: ", r_value_diff ** 2)
    x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
    y = np.asarray(np.mean(stim.X, axis=0)).ravel()
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    if verbose:
        print("All genes mean: ", r_value ** 2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
            # if "y1" in axis_keys.keys():
            # y1_bar = y1[j]
            # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
    if gene_list is not None:
        adjust_text(
            texts,
            x=x,
            y=y,
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
            force_points=(0.0, 0.0),
        )
    if legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if title is None:
        plt.title("", fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)
    ax.text(
        max(x) - max(x) * x_coeff,
        max(y) - y_coeff * max(y),
        r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " +
        f"{r_value ** 2:.2f}",
        fontsize=kwargs.get("textsize", fontsize),
    )
    if diff_genes is not None:
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - (y_coeff + 0.15) * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
            + f"{r_value_diff ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
    if save:
        plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    plt.close()
    if diff_genes is not None:
        return r_value ** 2, r_value_diff ** 2
    else:
        return r_value ** 2
