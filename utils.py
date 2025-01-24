from tabulate import tabulate
import numpy as np

def inputs_summary(data, args) -> str:
    columns = data.columns.to_list()
    rows = data.index.to_list()
    n_gt_data = np.sum(~np.isnan(data.to_numpy()))
    data_table = [
        [
            f"{data.shape[1]} datasets",
            f"""
                {', '.join(columns[:2])}, ... {', '.join(columns[-2:])}
            """
        ],
        [
            f"{data.shape[0]} models",
            f"""
                {', '.join(rows[:2])}, ... {', '.join(rows[-2:])}
            """
        ],
        [
            f"Ground truth data points",
            f"{n_gt_data} values"
        ]
    ]
    params_table = [
        [
            "Learning rate",
            args.lr
        ],
        [
            "Steps",
            args.n_iter
        ],
        [
            "Distance",
            args.dist
        ],
        [
            "Dimensions",
            args.dims
        ],
        [
            "Normalize",
            args.normalize
        ],
        [
            "Val size",
            args.val
        ],
        [
            "Use UMAP",
            not args.no_umap
        ]
    ]

    return tabulate(data_table, tablefmt='fancy_grid') + '\n' + tabulate(params_table, tablefmt='fancy_grid')

