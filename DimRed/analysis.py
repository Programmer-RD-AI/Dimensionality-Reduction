from DimRed import *


class Analysis:
    def __init__(self, X: np.array, y: np.array):
        self.X = X
        self.size_per_side = int(math.sqrt(len(self.X)))
        self.y = y

    def produce_combinations(
        self, name: str, standard_pipeline: Pipeline, pipeline: Pipeline
    ) -> None:
        standard_pipeline.fit(self.X, self.y)
        pipeline.fit(self.X, self.y)
        X_standard_embedded = standard_pipeline.transform(self.X)
        X_custom_pipeline_embedded = pipeline.transform(self.X)
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(f"{name}-{standard_pipeline}")
        ax_custom = (
            fig.add_subplot(111, projection="3d")
            if X_custom_pipeline_embedded.shape[-1] > 2
            else fig.add_subplot(111)
        )
        if X_custom_pipeline_embedded.shape[-1] == 1:
            x, y = (
                X_custom_pipeline_embedded[:, 0],
                np.zeros(len(X_custom_pipeline_embedded)),
            )
            custom_scatter = ax_custom.scatter(
                x,
                y,
                c=self.y,
                s=20,
                cmap="Set1",
                label="1D Scatter Plot",
            )
        elif X_custom_pipeline_embedded.shape[-1] == 2:
            x, y = (
                X_custom_pipeline_embedded[:, 0],
                X_custom_pipeline_embedded[:, 1],
            )
            custom_scatter = ax_custom.scatter(
                x, y, c=self.y, s=20, cmap="Set2", label="2D Scatter Plot"
            )
        else:
            x, y, z = (
                X_custom_pipeline_embedded[:, 0],
                X_custom_pipeline_embedded[:, 1],
                X_custom_pipeline_embedded[:, 2],
            )
            custom_scatter = ax_custom.scatter(
                x,
                y,
                z,
                s=20,  # Specify the size of the markers
                c=self.y,
                cmap="Set3",
                label="3D Scatter Plot",
            )
        ax_custom.set_title(f"{pipeline}", fontsize=10)
        color_bar_custom = plt.colorbar(custom_scatter, ax=ax_custom)
        color_bar_custom.set_label("Classes")
        plt.legend()
        dirs = director_exist(os.path.join(os.getenv("GRAPH_PATH"), run, name))
        sd_pipeline_name = str(standard_pipeline)
        sd_pipeline_name.strip
        cs_pipeline_name = str(pipeline)
        cs_pipeline_name.strip
        plt.savefig(f"{dirs}/{cs_pipeline_name}.png", dpi=300)
