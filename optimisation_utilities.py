import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def PrintPF(title, pf):
    is_series = isinstance(pf, pd.Series)
    is_df     = isinstance(pf, pd.DataFrame)
    print(title)
    print("----------------")
    if is_df:
        print(f"Return: {pf.Ret.iloc[0]:.4%}")
        print(f"Volatility: {pf.Vol.iloc[0]:.4%}")
        print(f"Ret/Vol: {pf["Ret/Vol"].iloc[0]:.2f}")
        print("   Composition")
        for stock, weight in pf.Comp.iloc[0].items():
            print(f"   {stock}: {weight:.2%}")
        print("")
    elif is_series:
        print(f"Return: {pf.Ret:.4%}")
        print(f"Volatility: {pf.Vol:.4%}")
        print(f"Ret/Vol: {pf["Ret/Vol"]:.2f}")
        print("   Composition")
        for stock, weight in pf.Comp.items():
            print(f"   {stock}: {weight:.2%}")
        print("")
    else:
        raise TypeError("\'pf\' object must be either pd.DataFrame or pd.Series")
    return None

def ShowComposition(title, pf):
    is_series = isinstance(pf, pd.Series)
    is_df     = isinstance(pf, pd.DataFrame)
    if is_df:
        all_labels = list(pf.Comp.iloc[0].keys())
        all_values = np.fromiter(pf.Comp.iloc[0].values(), dtype=float)
    elif is_series:
        all_labels = list(pf.Comp.keys())
        all_values = np.fromiter(pf.Comp.values(), dtype=float)
    nonzero_mask = all_values != 0
    values = all_values[nonzero_mask]
    labels = np.array(all_labels)[nonzero_mask]
    abs_values = np.abs(values)
    color_map = ['green' if v > 0 else 'red' for v in values]
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    def format_signed_weight(val):
        # Convert percentage to actual weight (with correct sign)
        total = np.sum(abs_values)
        fraction = val / 100 * total
        diffs = np.abs(abs_values - fraction)
        idx = np.argmin(diffs)
        return f'{values[idx]:+.1%}'  # always show sign

    wedges, texts, autotexts = ax.pie(
        abs_values,
        labels=labels,
        colors=color_map,
        autopct=format_signed_weight,
        startangle=90,
        counterclock=False
    )

    for text in autotexts:
        text.set_color('white')
        text.set_fontweight('bold')
    for wedge in wedges:
        wedge.set_linewidth(1.3)
        wedge.set_edgecolor("white")

    ax.set_title(title)
    return None

def ShowReturns(title, pf):
    is_series = isinstance(pf, pd.Series)
    is_df     = isinstance(pf, pd.DataFrame)

    #---- Series ----#
    if is_series:
        n = len(pf)
        values = pf.to_numpy()
        names = pf.index.to_list()
    #---- DataFrame ----#
    elif is_df:
        n, _ = pf.shape
        values = pf.Ret
        it_worked = False
        for x in ["Names", "Stocks", "Assets"]:
            try:
                names = pf[x]
                it_worked = True
            except Exception as e:
                None
        if not it_worked:
            names = pf.index.to_list()
    else:
        raise TypeError("\'pf\' object must be either pd.DataFrame or pd.Series")
    
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    ax.set_title(title)
    bars = ax.barh(y = names, width = values)
    for bar, value in zip(bars, values):
        xpos, ypos = bar.get_width(), bar.get_y() + bar.get_height()/2
        if value<0:
            ax.text(0, ypos, f"{value:+.2%}", va="center", color="red")
            bar.set_hatch("//")
            bar.set_edgecolor("white")
        if value>0:
            ax.text(xpos, ypos, f"{value:+.2%}", va="center", color="green")
    ax.set_xlim(min(values)-0.1, max(values) + 0.3)
    return None

def ShowEfficientFrontier(title, Efficient_Frontier, n_points):
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    ax.set_title(title)
    ax.plot(np.sqrt(Efficient_Frontier.Vol.iloc[:n_points]),Efficient_Frontier.Ret.iloc[:n_points], label="Efficient Frontier")
    mv_idx = np.argmin(Efficient_Frontier.Vol.iloc[:n_points])
    tang_idx = np.argmax(Efficient_Frontier["Ret/Vol"].iloc[:n_points])
    ax.plot(np.sqrt(Efficient_Frontier.Vol.iloc[mv_idx]),Efficient_Frontier.Ret.iloc[mv_idx], "bx", label="Minimum Variance portfolio")
    ax.plot(np.sqrt(Efficient_Frontier.Vol.iloc[tang_idx]),Efficient_Frontier.Ret.iloc[tang_idx], "rx", label="Tangency portfolio")
    if len(Efficient_Frontier)>n_points:
        for i in range(n_points, len(Efficient_Frontier)):
            asset = Efficient_Frontier.iloc[i]
            all_weights = np.fromiter(asset.Comp.values(), dtype=float)
            all_names = list(asset.Comp.keys())
            name_idx = np.argmax(all_weights)
            assert all_weights[name_idx]==1.
            
            ax.plot(np.sqrt(asset.Vol), asset.Ret, marker="x", color="black")
            ax.annotate(all_names[name_idx], (np.sqrt(asset.Vol), asset.Ret))
    ax.legend()
    return None
            
    